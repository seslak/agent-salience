"""Running statistics primitives."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Mapping, Union


@dataclass
class RunningStats:
    """Welford running moments for deterministic online variance."""

    count: int = 0
    mean: float = 0.0
    m2: float = 0.0

    def update(self, value: float) -> RunningStats:
        x = float(value)
        self.count += 1
        delta = x - self.mean
        self.mean += delta / self.count
        self.m2 += delta * (x - self.mean)
        return self

    @property
    def variance(self) -> float:
        if self.count < 2:
            return 0.0
        return max(0.0, self.m2 / (self.count - 1))

    @property
    def stddev(self) -> float:
        return math.sqrt(self.variance)

    def to_dict(self) -> Dict[str, Union[float, int]]:
        return {"count": int(self.count), "mean": float(self.mean), "m2": float(self.m2)}

    @classmethod
    def from_dict(cls, data: Mapping[str, Union[float, int]]) -> RunningStats:
        return cls(
            count=int(data.get("count", 0)),
            mean=float(data.get("mean", 0.0)),
            m2=float(data.get("m2", 0.0)),
        )


@dataclass
class EwmaStats:
    """EWMA mean and approximate variance."""

    alpha: float
    mean: float = 0.0
    variance: float = 0.0
    initialized: bool = False

    def __post_init__(self) -> None:
        self.alpha = float(self.alpha)
        if not (0.0 < self.alpha <= 1.0):
            raise ValueError("alpha must be in (0, 1]")
        self.mean = float(self.mean)
        self.variance = max(0.0, float(self.variance))
        self.initialized = bool(self.initialized)

    def update(self, value: float) -> EwmaStats:
        x = float(value)
        if not self.initialized:
            self.mean = x
            self.variance = 0.0
            self.initialized = True
            return self
        self.mean = self.alpha * x + (1.0 - self.alpha) * self.mean
        delta = x - self.mean
        self.variance = self.alpha * (delta * delta) + (1.0 - self.alpha) * self.variance
        if self.variance < 0.0:
            self.variance = 0.0
        return self

    @property
    def stddev(self) -> float:
        return math.sqrt(max(0.0, self.variance))

    def to_dict(self) -> Dict[str, Union[float, bool]]:
        return {
            "alpha": float(self.alpha),
            "mean": float(self.mean),
            "variance": float(self.variance),
            "initialized": bool(self.initialized),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Union[float, bool]]) -> EwmaStats:
        return cls(
            alpha=float(data.get("alpha", 0.5)),
            mean=float(data.get("mean", 0.0)),
            variance=float(data.get("variance", 0.0)),
            initialized=bool(data.get("initialized", False)),
        )
