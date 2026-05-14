"""Drift and novelty helpers."""

from __future__ import annotations

from typing import Sequence

from .scoring import signal_score


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def drift_score(anchor_text: str, new_text: str) -> float:
    """Drift is inverse salience between anchor and new text."""
    score = signal_score(anchor_text, new_text).final
    return _clamp(1.0 - score)


def novelty_score(text: str, reference_texts: Sequence[str]) -> float:
    """Novelty is inverse best-match salience against references."""
    if not reference_texts:
        return 1.0
    best_match = max(signal_score(text, reference).final for reference in reference_texts)
    return _clamp(1.0 - best_match)
