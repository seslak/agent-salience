"""Public API for agent_salience."""

from __future__ import annotations

from .drift import drift_score, novelty_score
from .loops import ActionEvent, LoopDecision, detect_repeated_target_loop, repetition_score
from .idf import IdfProfile, build_domain_idf_profiles, build_idf_profile, idf_cosine_similarity, idf_weighted_vector
from .scoring import SignalBreakdown, cosine_similarity, jaccard_similarity, signal_score
from .stats import EwmaStats, RunningStats
from .text import (
    TextSignature,
    build_text_signature,
    char_ngram_similarity,
    char_ngrams,
    expand_tokens_with_aliases,
    normalize_alias_map,
    normalize_text,
    shingle_hashes,
    stable_hash_hex,
    token_frequencies,
    token_prefix_overlap,
    token_shingles,
)
from .thresholds import AdaptiveThreshold, HysteresisThreshold, ThresholdDecision
from .triggers import SignalDecision, SignalTrigger

__version__ = "0.2.0"

__all__ = [
    "__version__",
    "normalize_text",
    "token_frequencies",
    "stable_hash_hex",
    "token_shingles",
    "shingle_hashes",
    "TextSignature",
    "build_text_signature",
    "char_ngrams",
    "char_ngram_similarity",
    "token_prefix_overlap",
    "normalize_alias_map",
    "expand_tokens_with_aliases",
    "IdfProfile",
    "build_idf_profile",
    "build_domain_idf_profiles",
    "idf_weighted_vector",
    "idf_cosine_similarity",
    "cosine_similarity",
    "jaccard_similarity",
    "signal_score",
    "SignalBreakdown",
    "RunningStats",
    "EwmaStats",
    "AdaptiveThreshold",
    "ThresholdDecision",
    "HysteresisThreshold",
    "SignalTrigger",
    "SignalDecision",
    "ActionEvent",
    "LoopDecision",
    "repetition_score",
    "detect_repeated_target_loop",
    "drift_score",
    "novelty_score",
]
