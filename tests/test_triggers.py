from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agent_salience.explain import explain_signal_decision, explain_threshold_decision
from agent_salience.stats import RunningStats
from agent_salience.thresholds import AdaptiveThreshold
from agent_salience.triggers import SignalTrigger


class TriggerTests(unittest.TestCase):
    def test_signal_trigger_evaluate_does_not_mutate(self) -> None:
        threshold = AdaptiveThreshold(base=0.6, stats=RunningStats())
        trigger = SignalTrigger(name="test", pattern="alpha beta", threshold=threshold)
        before = threshold.stats.count if isinstance(threshold.stats, RunningStats) else -1
        decision = trigger.evaluate("alpha beta")
        after = threshold.stats.count if isinstance(threshold.stats, RunningStats) else -1
        self.assertEqual(before, after)
        self.assertGreaterEqual(decision.score, 0.0)

    def test_signal_trigger_observe_mutates(self) -> None:
        threshold = AdaptiveThreshold(base=0.6, stats=RunningStats())
        trigger = SignalTrigger(name="test", pattern="alpha beta", threshold=threshold, kind="resonance")
        trigger.observe("alpha beta")
        self.assertIsInstance(threshold.stats, RunningStats)
        assert isinstance(threshold.stats, RunningStats)
        self.assertEqual(threshold.stats.count, 1)

    def test_explain_helpers_non_empty(self) -> None:
        threshold = AdaptiveThreshold(base=0.6, stats=RunningStats())
        trigger = SignalTrigger(name="test", pattern="alpha", threshold=threshold)
        signal_decision = trigger.evaluate("alpha")
        threshold_decision = threshold.decide(signal_decision.score)
        self.assertTrue(explain_signal_decision(signal_decision))
        self.assertTrue(explain_threshold_decision(threshold_decision))

    def test_signal_decision_to_dict(self) -> None:
        threshold = AdaptiveThreshold(base=0.6, stats=RunningStats())
        trigger = SignalTrigger(name="test", pattern="alpha", threshold=threshold)
        signal_decision = trigger.evaluate("alpha")
        payload = signal_decision.to_dict()
        self.assertIn("breakdown", payload)
        self.assertIn("score", payload)


if __name__ == "__main__":
    unittest.main()
