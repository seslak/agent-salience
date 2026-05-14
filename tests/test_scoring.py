from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agent_salience.scoring import (
    SignalBreakdown,
    cosine_similarity,
    jaccard_similarity,
    signal_score,
)


class ScoringTests(unittest.TestCase):
    def test_cosine_similarity(self) -> None:
        score = cosine_similarity({"a": 1.0, "b": 1.0}, {"a": 1.0, "b": 1.0})
        self.assertAlmostEqual(score, 1.0, places=6)

    def test_cosine_similarity_empty(self) -> None:
        self.assertEqual(cosine_similarity({}, {"a": 1.0}), 0.0)

    def test_jaccard_similarity(self) -> None:
        score = jaccard_similarity(["a", "b"], ["b", "c"])
        self.assertAlmostEqual(score, 1.0 / 3.0, places=6)

    def test_jaccard_similarity_both_empty(self) -> None:
        self.assertEqual(jaccard_similarity([], []), 0.0)

    def test_signal_score_returns_breakdown(self) -> None:
        result = signal_score("alpha beta", "alpha gamma", repetition=0.2, recency=0.3)
        self.assertIsInstance(result, SignalBreakdown)
        self.assertGreaterEqual(result.final, 0.0)
        self.assertLessEqual(result.final, 1.0)
        self.assertIn("cosine", result.weights)

    def test_signal_score_exact_match_is_high_by_default(self) -> None:
        result = signal_score("alpha beta", "alpha beta")
        self.assertAlmostEqual(result.final, 1.0, places=6)

    def test_optional_components_need_nonzero_weights_to_affect_score(self) -> None:
        baseline = signal_score("alpha beta", "alpha beta", repetition=0.0)
        unchanged = signal_score("alpha beta", "alpha beta", repetition=1.0)
        self.assertAlmostEqual(baseline.final, unchanged.final, places=6)

        weighted = signal_score(
            "alpha beta",
            "alpha beta",
            repetition=0.0,
            weights={"cosine": 0.5, "jaccard": 0.3, "repetition": 0.2},
        )
        weighted_with_repetition = signal_score(
            "alpha beta",
            "alpha beta",
            repetition=1.0,
            weights={"cosine": 0.5, "jaccard": 0.3, "repetition": 0.2},
        )
        self.assertGreater(weighted_with_repetition.final, weighted.final)

    def test_signal_score_weight_normalization(self) -> None:
        result = signal_score("alpha beta", "alpha beta", weights={"cosine": 2.0, "jaccard": 2.0})
        self.assertAlmostEqual(result.weights["cosine"], 0.5, places=6)
        self.assertAlmostEqual(result.weights["jaccard"], 0.5, places=6)
        self.assertEqual(result.weights["repetition"], 0.0)
        self.assertEqual(result.weights["recency"], 0.0)

    def test_signal_breakdown_to_dict(self) -> None:
        result = signal_score("alpha beta", "alpha beta")
        payload = result.to_dict()
        self.assertIsInstance(payload, dict)
        self.assertIn("final", payload)
        self.assertIn("weights", payload)


if __name__ == "__main__":
    unittest.main()


class ExtendedScoringTests(unittest.TestCase):
    def test_signal_score_default_remains_lexical(self) -> None:
        result = signal_score("alpha beta", "alpha beta")
        self.assertAlmostEqual(result.final, 1.0, places=6)
        self.assertEqual(result.char_ngram, 0.0)
        self.assertEqual(result.prefix, 0.0)
        self.assertEqual(result.idf_status, "not_requested")
        self.assertFalse(result.idf_used)

    def test_signal_score_with_fuzzy_weights(self) -> None:
        result = signal_score(
            "validation failure",
            "validated failures",
            include_fuzzy=True,
            weights={"cosine": 0.3, "jaccard": 0.2, "char_ngram": 0.3, "prefix": 0.2},
        )
        self.assertGreater(result.char_ngram, 0.0)
        self.assertGreater(result.prefix, 0.0)
        self.assertGreater(result.final, 0.0)

    def test_signal_score_with_alias_map_solves_semantic_gap(self) -> None:
        aliases = {"test_failure": ["test failure", "broken validation", "debug failing test"]}
        without_alias = signal_score("test failure triage", "debug broken validation run")
        with_alias = signal_score(
            "test failure triage",
            "debug broken validation run",
            alias_map=aliases,
            weights={"cosine": 0.4, "jaccard": 0.3, "alias": 0.3},
        )
        self.assertGreater(with_alias.final, without_alias.final)
        self.assertGreater(with_alias.alias, 0.0)

    def test_signal_score_auto_idf_reports_cold_fallback(self) -> None:
        from agent_salience.idf import build_idf_profile

        profile = build_idf_profile(["alpha beta"], min_documents=10)
        result = signal_score("alpha", "alpha", mode="auto", idf_profile=profile)
        self.assertEqual(result.idf_status, "cold")
        self.assertFalse(result.idf_used)
        self.assertEqual(result.idf_cosine, 0.0)
        self.assertGreater(result.final, 0.0)

    def test_signal_score_idf_ready_component(self) -> None:
        from agent_salience.idf import build_idf_profile

        profile = build_idf_profile(
            ["alpha beta", "alpha gamma", "delta epsilon"],
            min_documents=3,
            min_unique_terms=5,
            min_total_tokens=6,
        )
        result = signal_score(
            "alpha beta",
            "alpha gamma",
            mode="auto",
            idf_profile=profile,
            weights={"idf_cosine": 1.0},
        )
        self.assertEqual(result.idf_status, "ready")
        self.assertTrue(result.idf_used)
        self.assertGreater(result.final, 0.0)
