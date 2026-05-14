"""Compact explanation helpers for structured responses and logs."""

from __future__ import annotations

from .loops import LoopDecision
from .thresholds import ThresholdDecision
from .triggers import SignalDecision


def explain_threshold_decision(decision: ThresholdDecision) -> str:
    state = "TRIGGERED" if decision.triggered else "NOT_TRIGGERED"
    return (
        f"{state}: value={decision.value:.3f}, threshold={decision.threshold:.3f}, "
        f"margin={decision.margin:.3f}; {decision.reason}"
    )


def explain_signal_decision(decision: SignalDecision) -> str:
    state = "TRIGGERED" if decision.triggered else "NOT_TRIGGERED"
    return (
        f"{decision.name} ({decision.kind}) {state}: score={decision.score:.3f}, "
        f"threshold={decision.threshold:.3f}, margin={decision.margin:.3f}; {decision.reason}"
    )


def explain_loop_decision(decision: LoopDecision) -> str:
    state = "TRIGGERED" if decision.triggered else "NOT_TRIGGERED"
    return (
        f"{state}: repeated_count={decision.repeated_count}, "
        f"unique_targets={decision.unique_targets}, score={decision.repetition_score:.3f}; "
        f"{decision.reason}"
    )
