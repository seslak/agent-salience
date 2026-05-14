"""Threshold primitives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Tuple, Union

from .stats import EwmaStats, RunningStats


def _clamp(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    return max(minimum, min(maximum, value))


@dataclass(frozen=True)
class ThresholdDecision:
    value: float
    threshold: float
    triggered: bool
    margin: float
    reason: str

    def to_dict(self) -> Dict[str, Union[float, bool, str]]:
        return {
            "value": float(self.value),
            "threshold": float(self.threshold),
            "triggered": bool(self.triggered),
            "margin": float(self.margin),
            "reason": str(self.reason),
        }


@dataclass
class AdaptiveThreshold:
    base: float = 0.70
    minimum: float = 0.45
    maximum: float = 0.95
    k: float = 0.75
    stats: Optional[Union[EwmaStats, RunningStats]] = None

    def observe(self, value: float) -> None:
        observed = _clamp(float(value))
        if self.stats is None:
            self.stats = RunningStats()
        self.stats.update(observed)

    def _mean_stddev(self) -> Optional[Tuple[float, float]]:
        if self.stats is None:
            return None
        if isinstance(self.stats, RunningStats):
            if self.stats.count <= 0:
                return None
            return self.stats.mean, self.stats.stddev
        if isinstance(self.stats, EwmaStats):
            if not self.stats.initialized:
                return None
            return self.stats.mean, self.stats.stddev
        return None

    def current(self) -> float:
        point = self._mean_stddev()
        if point is None:
            return _clamp(float(self.base), float(self.minimum), float(self.maximum))
        mean, stddev = point
        raw = float(mean) + float(self.k) * float(stddev)
        return _clamp(raw, float(self.minimum), float(self.maximum))

    def decide(self, value: float) -> ThresholdDecision:
        score = _clamp(float(value))
        threshold = self.current()
        triggered = score >= threshold
        margin = score - threshold
        if triggered:
            reason = f"value {score:.3f} met threshold {threshold:.3f}"
        else:
            reason = f"value {score:.3f} below threshold {threshold:.3f}"
        return ThresholdDecision(
            value=score,
            threshold=threshold,
            triggered=triggered,
            margin=margin,
            reason=reason,
        )

    def to_dict(self) -> Dict[str, Any]:
        stats_type: Optional[str] = None
        stats_payload: Optional[Dict[str, Any]] = None
        if isinstance(self.stats, RunningStats):
            stats_type = "running"
            stats_payload = dict(self.stats.to_dict())
        elif isinstance(self.stats, EwmaStats):
            stats_type = "ewma"
            stats_payload = dict(self.stats.to_dict())
        return {
            "base": float(self.base),
            "minimum": float(self.minimum),
            "maximum": float(self.maximum),
            "k": float(self.k),
            "stats_type": stats_type,
            "stats": stats_payload,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> AdaptiveThreshold:
        stats_type = str(data.get("stats_type") or "")
        stats_payload = data.get("stats")
        stats: Optional[Union[EwmaStats, RunningStats]] = None
        if stats_type == "running" and isinstance(stats_payload, Mapping):
            stats = RunningStats.from_dict(stats_payload)
        elif stats_type == "ewma" and isinstance(stats_payload, Mapping):
            stats = EwmaStats.from_dict(stats_payload)
        return cls(
            base=float(data.get("base", 0.70)),
            minimum=float(data.get("minimum", 0.45)),
            maximum=float(data.get("maximum", 0.95)),
            k=float(data.get("k", 0.75)),
            stats=stats,
        )


@dataclass
class HysteresisThreshold:
    enter: float
    exit: float
    active: bool = False

    def __post_init__(self) -> None:
        self.enter = _clamp(float(self.enter))
        self.exit = _clamp(float(self.exit))
        if self.exit > self.enter:
            raise ValueError("exit must be <= enter")
        self.active = bool(self.active)

    def decide(self, value: float) -> ThresholdDecision:
        score = _clamp(float(value))
        was_active = self.active
        threshold = self.exit if was_active else self.enter

        if not was_active and score >= self.enter:
            self.active = True
            reason = f"value {score:.3f} crossed enter threshold {self.enter:.3f}"
        elif was_active and score < self.exit:
            self.active = False
            reason = f"value {score:.3f} dropped below exit threshold {self.exit:.3f}"
        elif self.active:
            reason = f"value {score:.3f} kept hysteresis active above exit {self.exit:.3f}"
        else:
            reason = f"value {score:.3f} below enter threshold {self.enter:.3f}"

        return ThresholdDecision(
            value=score,
            threshold=threshold,
            triggered=self.active,
            margin=score - threshold,
            reason=reason,
        )
