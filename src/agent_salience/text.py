"""Deterministic lexical normalization and signature helpers."""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Mapping, Optional, Sequence


_TOKEN_RE = re.compile(r"(?u)[^\W_]+")
NORMALIZER_VERSION = 1
SIGNATURE_VERSION = 1


def normalize_text(text: str) -> list[str]:
    """Normalize text into deterministic alphanumeric lowercase tokens."""
    if not text:
        return []
    lowered = text.lower()
    return [match.group(0) for match in _TOKEN_RE.finditer(lowered)]


def token_frequencies(text: str) -> dict[str, float]:
    """Count normalized token frequencies as floats."""
    tokens = normalize_text(text)
    if not tokens:
        return {}
    counts = Counter(tokens)
    # Stable ordering keeps deterministic dict iteration when serialized.
    return {token: float(count) for token, count in sorted(counts.items())}


def stable_hash_hex(value: str, *, digest_size: int = 8) -> str:
    """Return a deterministic blake2b hex digest.

    This deliberately does not use Python's built-in hash(), because built-in
    hashes are salted per process and cannot be persisted as stable signatures.
    """
    return hashlib.blake2b(str(value).encode("utf-8"), digest_size=int(digest_size)).hexdigest()


def token_shingles(tokens: Sequence[str], *, size: int = 3) -> list[tuple[str, ...]]:
    """Build word-token shingles.

    Empty/tiny inputs return an empty list rather than a synthetic shingle.
    This keeps "not enough evidence" distinct from "perfect similarity".
    """
    n = max(1, int(size))
    if len(tokens) < n:
        return []
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def shingle_hashes(tokens: Sequence[str], *, size: int = 3, max_hashes: int = 256) -> list[str]:
    """Return the smallest deterministic hashes for token shingles.

    Keeping the min-k hashes gives a compact, future-LSH-friendly signature
    without introducing probabilistic LSH buckets yet.
    """
    shingles = token_shingles(tokens, size=size)
    if not shingles:
        return []
    hashes = [stable_hash_hex(" ".join(shingle)) for shingle in shingles]
    hashes.sort()
    cap = max(1, int(max_hashes))
    return hashes[:cap]


def top_terms(tokens: Sequence[str], *, limit: int = 20) -> list[str]:
    """Return top tokens by frequency, with deterministic alpha tie-breaks."""
    if not tokens:
        return []
    counts = Counter(tokens)
    return sorted(counts.keys(), key=lambda token: (-counts[token], token))[: max(1, int(limit))]


def normalize_raw_text_for_hash(text: str) -> str:
    """Normalize line endings/trailing spaces for raw content hashing."""
    normalized = str(text).replace("\r\n", "\n").replace("\r", "\n")
    return "\n".join(line.rstrip() for line in normalized.split("\n"))


@dataclass(frozen=True)
class TextSignature:
    """Compact deterministic signature for future candidate indexing.

    This is not an LSH index. It is a small persisted shape that makes a later
    MinHash/LSH implementation a signature-version upgrade instead of a schema
    redesign.
    """

    content_hash: str
    normalized_hash: str
    token_count: int
    unique_token_count: int
    top_terms: list[str]
    shingle_hashes: list[str]
    signature_version: int = SIGNATURE_VERSION
    normalizer_version: int = NORMALIZER_VERSION

    def to_dict(self) -> dict[str, object]:
        return {
            "content_hash": self.content_hash,
            "normalized_hash": self.normalized_hash,
            "token_count": int(self.token_count),
            "unique_token_count": int(self.unique_token_count),
            "top_terms": list(self.top_terms),
            "shingle_hashes": list(self.shingle_hashes),
            "signature_version": int(self.signature_version),
            "normalizer_version": int(self.normalizer_version),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "TextSignature":
        return cls(
            content_hash=str(payload.get("content_hash", "")),
            normalized_hash=str(payload.get("normalized_hash", "")),
            token_count=int(payload.get("token_count", 0) or 0),
            unique_token_count=int(payload.get("unique_token_count", 0) or 0),
            top_terms=[str(item) for item in payload.get("top_terms", []) or []],
            shingle_hashes=[str(item) for item in payload.get("shingle_hashes", []) or []],
            signature_version=int(payload.get("signature_version", SIGNATURE_VERSION) or SIGNATURE_VERSION),
            normalizer_version=int(payload.get("normalizer_version", NORMALIZER_VERSION) or NORMALIZER_VERSION),
        )


def build_text_signature(
    text: str,
    *,
    max_chars: int = 50_000,
    shingle_size: int = 3,
    max_shingle_hashes: int = 256,
    top_terms_limit: int = 20,
) -> TextSignature:
    """Build a deterministic compact text signature.

    ``content_hash`` covers the full raw text after stable line-ending
    normalization. Token/shingle work is capped so pasted logs cannot create
    unbounded CPU work.
    """
    raw = normalize_raw_text_for_hash(text)
    capped = str(text)[: max(0, int(max_chars))]
    tokens = normalize_text(capped)
    return TextSignature(
        content_hash=stable_hash_hex(raw),
        normalized_hash=stable_hash_hex(" ".join(tokens)),
        token_count=len(tokens),
        unique_token_count=len(set(tokens)),
        top_terms=top_terms(tokens, limit=top_terms_limit),
        shingle_hashes=shingle_hashes(tokens, size=shingle_size, max_hashes=max_shingle_hashes),
    )


def char_ngrams(text: str, *, n: int = 3) -> set[str]:
    """Return language-agnostic character n-grams over normalized tokens."""
    size = max(1, int(n))
    grams: set[str] = set()
    for token in normalize_text(text):
        if len(token) < size:
            if token:
                grams.add(token)
            continue
        for idx in range(len(token) - size + 1):
            grams.add(token[idx : idx + size])
    return grams


def char_ngram_similarity(source_text: str, target_text: str, *, n: int = 3) -> float:
    """Jaccard similarity over character n-grams.

    Useful for morphological variants such as validate/validated/validation;
    not intended to solve semantic synonyms.
    """
    left = char_ngrams(source_text, n=n)
    right = char_ngrams(target_text, n=n)
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def token_prefixes(tokens: Iterable[str], *, min_prefix: int = 5) -> set[str]:
    """Return deterministic token prefixes for fuzzy lexical family overlap."""
    size = max(1, int(min_prefix))
    prefixes: set[str] = set()
    for token in tokens:
        token = str(token).lower()
        if len(token) >= size:
            prefixes.add(token[:size])
    return prefixes


def token_prefix_overlap(
    source_tokens: Iterable[str],
    target_tokens: Iterable[str],
    *,
    min_prefix: int = 5,
) -> float:
    """Jaccard similarity over token prefixes.

    This is not alias expansion and not stemming. It is a small deterministic
    helper for visible word-family overlap such as config/configuration.
    """
    left = token_prefixes(source_tokens, min_prefix=min_prefix)
    right = token_prefixes(target_tokens, min_prefix=min_prefix)
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _phrase_key(tokens: Sequence[str]) -> str:
    return "_".join(tokens)


def normalize_alias_map(alias_map: Optional[Mapping[str, Iterable[str]]]) -> dict[str, list[list[str]]]:
    """Normalize a project/user alias map.

    Input shape is canonical term -> list of aliases/phrases. The canonical key
    itself is also treated as an alias phrase.
    """
    if not alias_map:
        return {}
    normalized: dict[str, list[list[str]]] = {}
    for canonical, aliases in alias_map.items():
        canonical_tokens = normalize_text(str(canonical))
        if not canonical_tokens:
            continue
        canonical_key = _phrase_key(canonical_tokens)
        phrases: list[list[str]] = []
        seen: set[tuple[str, ...]] = set()
        for phrase in [str(canonical), *[str(item) for item in aliases or []]]:
            phrase_tokens = normalize_text(phrase)
            if not phrase_tokens:
                continue
            key = tuple(phrase_tokens)
            if key in seen:
                continue
            seen.add(key)
            phrases.append(phrase_tokens)
        normalized[canonical_key] = phrases
    return normalized


def _contains_phrase(tokens: Sequence[str], phrase: Sequence[str]) -> bool:
    if not phrase or len(phrase) > len(tokens):
        return False
    plen = len(phrase)
    for idx in range(len(tokens) - plen + 1):
        if list(tokens[idx : idx + plen]) == list(phrase):
            return True
    return False


def expand_tokens_with_aliases(
    tokens: Sequence[str],
    alias_map: Optional[Mapping[str, Iterable[str]]],
) -> list[str]:
    """Return tokens expanded with canonical project aliases.

    Alias maintenance is intentionally external policy. This helper only applies
    an approved alias map supplied by the caller.
    """
    normalized_tokens = [str(token).lower() for token in tokens]
    expanded = list(normalized_tokens)
    normalized_aliases = normalize_alias_map(alias_map)
    if not normalized_aliases:
        return expanded
    for canonical_key, phrases in normalized_aliases.items():
        for phrase in phrases:
            if _contains_phrase(normalized_tokens, phrase):
                expanded.append(canonical_key)
                break
    return expanded
