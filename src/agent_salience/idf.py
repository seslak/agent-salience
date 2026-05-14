"""Project-local IDF helpers.

IDF support is intentionally cold-start aware. It should be learned from the
local project corpus and used only after a maturity threshold is reached.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Mapping, Optional, Sequence, Union

from .text import expand_tokens_with_aliases, normalize_text


def _cosine_similarity(a: Mapping[str, float], b: Mapping[str, float]) -> float:
    if not a or not b:
        return 0.0
    norm_a = math.sqrt(sum(float(value) * float(value) for value in a.values()))
    norm_b = math.sqrt(sum(float(value) * float(value) for value in b.values()))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    dot = sum(float(a[key]) * float(b[key]) for key in a.keys() & b.keys())
    return max(0.0, min(1.0, dot / (norm_a * norm_b)))


@dataclass(frozen=True)
class IdfProfile:
    domain: Optional[str]
    doc_count: int
    total_tokens: int
    unique_terms: int
    idf: dict[str, float]
    status: str
    min_documents: int
    min_unique_terms: int
    min_total_tokens: int
    version: int = 1

    @property
    def ready(self) -> bool:
        return self.status == "ready"

    def to_dict(self) -> dict[str, object]:
        return {
            "domain": self.domain,
            "doc_count": int(self.doc_count),
            "total_tokens": int(self.total_tokens),
            "unique_terms": int(self.unique_terms),
            "status": self.status,
            "ready": self.ready,
            "idf": {key: float(value) for key, value in sorted(self.idf.items())},
            "min_documents": int(self.min_documents),
            "min_unique_terms": int(self.min_unique_terms),
            "min_total_tokens": int(self.min_total_tokens),
            "version": int(self.version),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "IdfProfile":
        raw_idf = payload.get("idf", {}) or {}
        return cls(
            domain=None if payload.get("domain") is None else str(payload.get("domain")),
            doc_count=int(payload.get("doc_count", 0) or 0),
            total_tokens=int(payload.get("total_tokens", 0) or 0),
            unique_terms=int(payload.get("unique_terms", 0) or 0),
            idf={str(k): float(v) for k, v in dict(raw_idf).items()},
            status=str(payload.get("status", "cold")),
            min_documents=int(payload.get("min_documents", 200) or 200),
            min_unique_terms=int(payload.get("min_unique_terms", 1000) or 1000),
            min_total_tokens=int(payload.get("min_total_tokens", 10000) or 10000),
            version=int(payload.get("version", 1) or 1),
        )


def build_idf_profile(
    documents: Iterable[str],
    *,
    domain: Optional[str] = None,
    min_documents: int = 200,
    min_unique_terms: int = 1000,
    min_total_tokens: int = 10000,
) -> IdfProfile:
    """Build a local corpus IDF profile with maturity gating."""
    doc_freq: Counter[str] = Counter()
    doc_count = 0
    total_tokens = 0
    for document in documents:
        tokens = normalize_text(str(document))
        if not tokens:
            continue
        doc_count += 1
        total_tokens += len(tokens)
        doc_freq.update(set(tokens))
    unique_terms = len(doc_freq)
    status = (
        "ready"
        if doc_count >= min_documents
        and unique_terms >= min_unique_terms
        and total_tokens >= min_total_tokens
        else "cold"
    )
    # Smooth formula. It is still useful to serialize the values while cold for
    # diagnostics, but callers should not use the profile unless status=ready.
    idf = {
        term: math.log((1.0 + doc_count) / (1.0 + freq)) + 1.0
        for term, freq in sorted(doc_freq.items())
    }
    return IdfProfile(
        domain=domain,
        doc_count=doc_count,
        total_tokens=total_tokens,
        unique_terms=unique_terms,
        idf=idf,
        status=status,
        min_documents=int(min_documents),
        min_unique_terms=int(min_unique_terms),
        min_total_tokens=int(min_total_tokens),
    )


def build_domain_idf_profiles(
    records: Iterable[Mapping[str, object]],
    *,
    text_key: str = "text",
    domain_key: str = "domain",
    min_documents: int = 200,
    min_unique_terms: int = 1000,
    min_total_tokens: int = 10000,
) -> dict[str, IdfProfile]:
    """Build one IDF profile per domain from structured local records."""
    grouped: dict[str, list[str]] = {}
    for record in records:
        domain = str(record.get(domain_key) or "default")
        grouped.setdefault(domain, []).append(str(record.get(text_key) or ""))
    return {
        domain: build_idf_profile(
            docs,
            domain=domain,
            min_documents=min_documents,
            min_unique_terms=min_unique_terms,
            min_total_tokens=min_total_tokens,
        )
        for domain, docs in sorted(grouped.items())
    }


def idf_weighted_vector(
    text: str,
    profile: Union[IdfProfile, Mapping[str, object]],
    *,
    alias_map: Optional[Mapping[str, Sequence[str]]] = None,
) -> dict[str, float]:
    """Return TF-IDF-like sparse vector if the profile is ready."""
    if not isinstance(profile, IdfProfile):
        profile = IdfProfile.from_dict(profile)
    if not profile.ready:
        return {}
    tokens = normalize_text(text)
    tokens = expand_tokens_with_aliases(tokens, alias_map)
    if not tokens:
        return {}
    counts = Counter(tokens)
    return {
        token: float(count) * float(profile.idf.get(token, 1.0))
        for token, count in sorted(counts.items())
    }


def idf_cosine_similarity(
    source_text: str,
    target_text: str,
    profile: Union[IdfProfile, Mapping[str, object]],
    *,
    alias_map: Optional[Mapping[str, Sequence[str]]] = None,
) -> float:
    """Cosine similarity over IDF-weighted local corpus vectors.

    Returns 0.0 while the profile is cold. Callers can then fall back to lexical
    scoring and report idf_status/idf_used in diagnostics.
    """
    left = idf_weighted_vector(source_text, profile, alias_map=alias_map)
    right = idf_weighted_vector(target_text, profile, alias_map=alias_map)
    return _cosine_similarity(left, right)
