from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agent_salience.explain import explain_loop_decision
from agent_salience.loops import ActionEvent, detect_repeated_target_loop, repetition_score


class LoopTests(unittest.TestCase):
    def test_repetition_score_empty(self) -> None:
        self.assertEqual(repetition_score([]), 0.0)

    def test_repetition_score_filtered(self) -> None:
        events = [
            ActionEvent(tool="file_window", target="a.py"),
            ActionEvent(tool="file_window", target="a.py"),
            ActionEvent(tool="rank_files", target="."),
        ]
        score = repetition_score(events, tool="file_window")
        self.assertAlmostEqual(score, 2.0 / 3.0, places=6)

    def test_detect_repeated_target_loop(self) -> None:
        events = [ActionEvent(tool="file_window", target="a.py") for _ in range(6)]
        decision = detect_repeated_target_loop(events, threshold=0.6, min_count=5)
        self.assertTrue(decision.triggered)
        self.assertEqual(decision.repeated_count, 6)

    def test_action_event_to_dict(self) -> None:
        event = ActionEvent(tool="file_window", target="a.py", detail="line 1", tokens=10)
        payload = event.to_dict()
        self.assertEqual(payload["tool"], "file_window")
        loaded = ActionEvent.from_dict(payload)
        self.assertEqual(loaded.tool, event.tool)
        self.assertEqual(loaded.target, event.target)

    def test_loop_decision_to_dict(self) -> None:
        decision = detect_repeated_target_loop([ActionEvent(tool="a", target="b")], threshold=0.5, min_count=1)
        payload = decision.to_dict()
        self.assertIn("repetition_score", payload)
        self.assertIn("triggered", payload)

    def test_explain_loop_non_empty(self) -> None:
        decision = detect_repeated_target_loop([], threshold=0.6, min_count=5)
        self.assertTrue(explain_loop_decision(decision))


if __name__ == "__main__":
    unittest.main()
