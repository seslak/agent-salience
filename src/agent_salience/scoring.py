"""Signal scoring primitives."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Optional, Sequence, Union

from .idf import IdfProfile, idf_cosine_similarity
from .text import (
    char_ngram_similarity,
    expand_tokens_with_aliases,
    normalize_text,
    token_frequencies,
    token_prefix_overlap,
)


_COMPONENT_KEYS = (
    "cosine",
    "jaccard",
    "char_ngram",
    "prefix",
    "alias",
    "idf_cosine",
    "repetition",
    "recency",
    "novelty",
    "drift",
)
_DEFAULT_WEIGHTS: dict[str, float] = {
    "cosine": 0.70,
    "jaccard": 0.30,
    "char_ngram": 0.00,
    "prefix": 0.00,
    "alias": 0.00,
    "idf_cosine": 0.00,
    "repetition": 0.00,
    "recency": 0.00,
    "novelty": 0.00,
    "drift": 0.00,
}


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def cosine_similarity(a: Mapping[str, float], b: Mapping[str, float]) -> float:
    """Compute cosine similarity for sparse vectors."""
    if not a or not b:
        return 0.0
    norm_a = math.sqrt(sum(float(value) * float(value) for value in a.values()))
    norm_b = math.sqrt(sum(float(value) * float(value) for value in b.values()))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    dot = sum(float(a[key]) * float(b[key]) for key in a.keys() & b.keys())
    return _clamp(dot / (norm_a * norm_b))


def jaccard_similarity(a: Iterable[str], b: Iterable[str]) -> float:
    """Compute Jaccard similarity of two iterables.

    Empty-empty returns 0.0 by design: in search/memory salience, empty text is
    no evidence, not a perfect semantic match.
    """
    set_a = set(a)
    set_b = set(b)
    if not set_a and not set_b:
        return 0.0
    union = set_a | set_b
    if not union:
        return 0.0
    return _clamp(len(set_a & set_b) / len(union))


@dataclass(frozen=True)
class SignalBreakdown:
    cosine: float
    jaccard: float
    repetition: float
    recency: float
    novelty: float
    drift: float
    final: float
    weights: dict[str, float]
    char_ngram: float = 0.0
    prefix: float = 0.0
    alias: float = 0.0
    idf_cosine: float = 0.0
    idf_status: str = "not_requested"
    idf_used: bool = False

    def to_dict(self) -> Dict[str, Union[float, bool, str, Dict[str, float]]]:
        return {
            "cosine": float(self.cosine),
            "jaccard": float(self.jaccard),
            "char_ngram": float(self.char_ngram),
            "prefix": float(self.prefix),
            "alias": float(self.alias),
            "idf_cosine": float(self.idf_cosine),
            "repetition": float(self.repetition),
            "recency": float(self.recency),
            "novelty": float(self.novelty),
            "drift": float(self.drift),
            "final": float(self.final),
            "weights": {key: float(value) for key, value in self.weights.items()},
            "idf_status": self.idf_status,
            "idf_used": bool(self.idf_used),
        }


def _normalize_weights(weights: Optional[Mapping[str, float]]) -> dict[str, float]:
    if weights is None:
        base = {key: max(0.0, float(value)) for key, value in _DEFAULT_WEIGHTS.items()}
    else:
        base = {key: max(0.0, float(weights.get(key, 0.0))) for key in _COMPONENT_KEYS}
    total = sum(base.values())
    if total <= 0.0:
        return {key: 0.0 for key in _COMPONENT_KEYS}
    return {key: value / total for key, value in base.items()}


def _idf_status(profile: Optional[Union[IdfProfile, Mapping[str, object]]], mode: str) -> tuple[str, bool]:
    if mode not in {"idf", "auto"}:
        return "not_requested", False
    if profile is None:
        return "missing", False
    if not isinstance(profile, IdfProfile):
        profile = IdfProfile.from_dict(profile)
    return profile.status, profile.ready


def signal_score(
    source_text: str,
    target_text: str,
    *,
    repetition: float = 0.0,
    recency: float = 0.0,
    novelty: float = 0.0,
    drift: float = 0.0,
    weights: Optional[Mapping[str, float]] = None,
    include_fuzzy: bool = False,
    alias_map: Optional[Mapping[str, Sequence[str]]] = None,
    idf_profile: Optional[Union[IdfProfile, Mapping[str, object]]] = None,
    mode: str = "lexical",
) -> SignalBreakdown:
    """Compute a weighted explainable signal score.

    Default behavior is the original lexical cosine+jaccard scorer. Optional
    fuzzy/alias/IDF components are explicit and report their status in the
    returned breakdown.
    """
    source_tokens = normalize_text(source_text)
    target_tokens = normalize_text(target_text)
    expanded_source_tokens = expand_tokens_with_aliases(source_tokens, alias_map)
    expanded_target_tokens = expand_tokens_with_aliases(target_tokens, alias_map)

    source_vector = token_frequencies(" ".join(expanded_source_tokens))
    target_vector = token_frequencies(" ".join(expanded_target_tokens))

    cosine = _clamp(cosine_similarity(source_vector, target_vector))
    jaccard = _clamp(jaccard_similarity(expanded_source_tokens, expanded_target_tokens))
    repetition = _clamp(float(repetition))
    recency = _clamp(float(recency))
    novelty = _clamp(float(novelty))
    drift = _clamp(float(drift))

    normalized_weights = _normalize_weights(weights)
    wants_fuzzy = include_fuzzy or normalized_weights.get("char_ngram", 0.0) > 0.0 or normalized_weights.get("prefix", 0.0) > 0.0
    char_ngram = _clamp(char_ngram_similarity(source_text, target_text)) if wants_fuzzy else 0.0
    prefix = _clamp(token_prefix_overlap(source_tokens, target_tokens)) if wants_fuzzy else 0.0

    alias = 0.0
    if alias_map:
        source_aliases = set(expanded_source_tokens) - set(source_tokens)
        target_aliases = set(expanded_target_tokens) - set(target_tokens)
        alias = _clamp(jaccard_similarity(source_aliases, target_aliases))

    status, idf_ready = _idf_status(idf_profile, mode)
    idf_used = False
    idf_cosine = 0.0
    if mode in {"idf", "auto"} and idf_profile is not None and idf_ready:
        idf_cosine = _clamp(idf_cosine_similarity(source_text, target_text, idf_profile, alias_map=alias_map))
        idf_used = True

    components = {
        "cosine": cosine,
        "jaccard": jaccard,
        "char_ngram": char_ngram,
        "prefix": prefix,
        "alias": alias,
        "idf_cosine": idf_cosine,
        "repetition": repetition,
        "recency": recency,
        "novelty": novelty,
        "drift": drift,
    }
    final = _clamp(sum(components[key] * normalized_weights[key] for key in _COMPONENT_KEYS))

    return SignalBreakdown(
        cosine=cosine,
        jaccard=jaccard,
        char_ngram=char_ngram,
        prefix=prefix,
        alias=alias,
        idf_cosine=idf_cosine,
        repetition=repetition,
        recency=recency,
        novelty=novelty,
        drift=drift,
        final=final,
        weights=normalized_weights,
        idf_status=status,
        idf_used=idf_used,
    )
