from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agent_salience.drift import drift_score, novelty_score


class DriftTests(unittest.TestCase):
    def test_drift_score_identical_text(self) -> None:
        self.assertAlmostEqual(drift_score("alpha beta", "alpha beta"), 0.0, places=6)

    def test_drift_score_bounded(self) -> None:
        score = drift_score("alpha beta", "gamma delta")
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_novelty_score_no_references(self) -> None:
        self.assertEqual(novelty_score("anything", []), 1.0)

    def test_novelty_score_against_same_reference(self) -> None:
        self.assertAlmostEqual(novelty_score("alpha beta", ["alpha beta"]), 0.0, places=6)


if __name__ == "__main__":
    unittest.main()
