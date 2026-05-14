from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agent_salience.stats import RunningStats
from agent_salience.thresholds import AdaptiveThreshold, HysteresisThreshold


class ThresholdTests(unittest.TestCase):
    def test_adaptive_threshold_current_without_stats(self) -> None:
        threshold = AdaptiveThreshold(base=0.7, minimum=0.4, maximum=0.9)
        self.assertAlmostEqual(threshold.current(), 0.7, places=6)

    def test_adaptive_threshold_observe_and_decide(self) -> None:
        threshold = AdaptiveThreshold(base=0.7, minimum=0.4, maximum=0.95, k=0.5, stats=RunningStats())
        threshold.observe(0.8)
        threshold.observe(0.9)
        decision = threshold.decide(0.85)
        self.assertGreaterEqual(decision.threshold, threshold.minimum)
        self.assertLessEqual(decision.threshold, threshold.maximum)
        self.assertTrue(decision.reason)

    def test_threshold_decision_to_dict(self) -> None:
        threshold = AdaptiveThreshold(base=0.7, minimum=0.4, maximum=0.95)
        decision = threshold.decide(0.9)
        payload = decision.to_dict()
        self.assertIn("triggered", payload)
        self.assertIn("reason", payload)

    def test_adaptive_threshold_to_from_dict(self) -> None:
        threshold = AdaptiveThreshold(stats=RunningStats())
        threshold.observe(0.75)
        loaded = AdaptiveThreshold.from_dict(threshold.to_dict())
        self.assertAlmostEqual(loaded.current(), threshold.current(), places=6)

    def test_hysteresis_enter_exit_behavior(self) -> None:
        hysteresis = HysteresisThreshold(enter=0.7, exit=0.5)
        first = hysteresis.decide(0.75)
        self.assertTrue(first.triggered)
        second = hysteresis.decide(0.60)
        self.assertTrue(second.triggered)
        third = hysteresis.decide(0.40)
        self.assertFalse(third.triggered)


if __name__ == "__main__":
    unittest.main()
