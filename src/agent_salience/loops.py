"""Repeated-action loop diagnostics."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Sequence, Union


def _clamp(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    return max(minimum, min(maximum, value))


@dataclass(frozen=True)
class ActionEvent:
    tool: str
    target: str
    detail: str = ""
    tokens: int = 0

    def to_dict(self) -> Dict[str, Union[str, int]]:
        return {
            "tool": str(self.tool),
            "target": str(self.target),
            "detail": str(self.detail),
            "tokens": int(self.tokens),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> ActionEvent:
        return cls(
            tool=str(data.get("tool", "")),
            target=str(data.get("target", "")),
            detail=str(data.get("detail", "")),
            tokens=int(data.get("tokens", 0)),
        )


@dataclass(frozen=True)
class LoopDecision:
    repeated_count: int
    unique_targets: int
    repetition_score: float
    triggered: bool
    reason: str

    def to_dict(self) -> Dict[str, Union[int, float, bool, str]]:
        return {
            "repeated_count": int(self.repeated_count),
            "unique_targets": int(self.unique_targets),
            "repetition_score": float(self.repetition_score),
            "triggered": bool(self.triggered),
            "reason": str(self.reason),
        }


def repetition_score(
    events: Sequence[ActionEvent],
    *,
    tool: Optional[str] = None,
    target: Optional[str] = None,
) -> float:
    items = list(events)
    if not items:
        return 0.0

    if tool is None and target is None:
        pair_counts = Counter((event.tool, event.target) for event in items)
        matching = max(pair_counts.values(), default=0)
    else:
        matching = 0
        for event in items:
            if tool is not None and event.tool != tool:
                continue
            if target is not None and event.target != target:
                continue
            matching += 1
    return _clamp(matching / len(items))


def detect_repeated_target_loop(
    events: Sequence[ActionEvent],
    *,
    threshold: float = 0.60,
    min_count: int = 5,
) -> LoopDecision:
    items = list(events)
    if not items:
        return LoopDecision(
            repeated_count=0,
            unique_targets=0,
            repetition_score=0.0,
            triggered=False,
            reason="no events to evaluate",
        )

    pair_counts = Counter((event.tool, event.target) for event in items)
    (top_tool, top_target), repeated = pair_counts.most_common(1)[0]
    unique_targets = len(pair_counts)
    score = _clamp(repeated / len(items))
    threshold_value = _clamp(float(threshold))
    triggered = repeated >= int(min_count) and score >= threshold_value

    if triggered:
        reason = (
            f"repeated target loop detected for {top_tool}:{top_target} "
            f"({repeated}/{len(items)} events, score {score:.3f})"
        )
    else:
        reason = (
            f"no loop trigger: top repetition {repeated}/{len(items)} for "
            f"{top_tool}:{top_target}, score {score:.3f}"
        )

    return LoopDecision(
        repeated_count=repeated,
        unique_targets=unique_targets,
        repetition_score=score,
        triggered=triggered,
        reason=reason,
    )
