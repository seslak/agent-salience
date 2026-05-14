from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agent_salience.stats import EwmaStats, RunningStats


class StatsTests(unittest.TestCase):
    def test_running_stats_mean_variance(self) -> None:
        stats = RunningStats()
        stats.update(1.0).update(2.0).update(3.0)
        self.assertEqual(stats.count, 3)
        self.assertAlmostEqual(stats.mean, 2.0, places=6)
        self.assertAlmostEqual(stats.variance, 1.0, places=6)

    def test_running_stats_to_from_dict(self) -> None:
        stats = RunningStats().update(1.0).update(2.0)
        loaded = RunningStats.from_dict(stats.to_dict())
        self.assertEqual(loaded.count, stats.count)
        self.assertAlmostEqual(loaded.mean, stats.mean, places=6)
        self.assertAlmostEqual(loaded.m2, stats.m2, places=6)

    def test_ewma_update_behavior(self) -> None:
        stats = EwmaStats(alpha=0.5)
        stats.update(0.0)
        self.assertTrue(stats.initialized)
        self.assertAlmostEqual(stats.mean, 0.0, places=6)
        self.assertAlmostEqual(stats.variance, 0.0, places=6)

        stats.update(1.0)
        self.assertAlmostEqual(stats.mean, 0.5, places=6)
        self.assertAlmostEqual(stats.variance, 0.125, places=6)
        self.assertGreaterEqual(stats.stddev, 0.0)

    def test_invalid_ewma_alpha(self) -> None:
        with self.assertRaises(ValueError):
            EwmaStats(alpha=0.0)


if __name__ == "__main__":
    unittest.main()
