"""Trigger primitives built on salience scoring."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Union

from .scoring import SignalBreakdown, signal_score
from .thresholds import AdaptiveThreshold


@dataclass(frozen=True)
class SignalDecision:
    name: str
    kind: str
    score: float
    threshold: float
    triggered: bool
    margin: float
    breakdown: SignalBreakdown
    reason: str

    def to_dict(self) -> Dict[str, Union[str, float, bool, dict]]:
        return {
            "name": str(self.name),
            "kind": str(self.kind),
            "score": float(self.score),
            "threshold": float(self.threshold),
            "triggered": bool(self.triggered),
            "margin": float(self.margin),
            "breakdown": self.breakdown.to_dict(),
            "reason": str(self.reason),
        }


@dataclass
class SignalTrigger:
    """Kind metadata examples: resonance, anti_resonance, drift, loop, novelty, invariant.

    This library does not hard-code semantics by kind. For example, anti_resonance can
    represent caller-defined salience for undesirable patterns.
    """

    name: str
    pattern: str
    threshold: AdaptiveThreshold
    kind: str = "signal"

    def evaluate(self, text: str) -> SignalDecision:
        breakdown = signal_score(self.pattern, text)
        threshold_value = self.threshold.current()
        triggered = breakdown.final >= threshold_value
        margin = breakdown.final - threshold_value
        if triggered:
            reason = f"score {breakdown.final:.3f} met threshold {threshold_value:.3f}"
        else:
            reason = f"score {breakdown.final:.3f} below threshold {threshold_value:.3f}"
        return SignalDecision(
            name=self.name,
            kind=self.kind,
            score=breakdown.final,
            threshold=threshold_value,
            triggered=triggered,
            margin=margin,
            breakdown=breakdown,
            reason=reason,
        )

    def observe(self, text: str) -> SignalDecision:
        decision = self.evaluate(text)
        self.threshold.observe(decision.score)
        updated_threshold = self.threshold.current()
        reason = (
            f"{decision.reason}; observed score and updated threshold baseline "
            f"to {updated_threshold:.3f}"
        )
        return SignalDecision(
            name=decision.name,
            kind=decision.kind,
            score=decision.score,
            threshold=decision.threshold,
            triggered=decision.triggered,
            margin=decision.margin,
            breakdown=decision.breakdown,
            reason=reason,
        )
