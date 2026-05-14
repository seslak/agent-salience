from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agent_salience.idf import (
    IdfProfile,
    build_domain_idf_profiles,
    build_idf_profile,
    idf_cosine_similarity,
    idf_weighted_vector,
)


class IdfTests(unittest.TestCase):
    def test_idf_profile_is_cold_until_thresholds_met(self) -> None:
        profile = build_idf_profile(["alpha beta", "beta gamma"], min_documents=10)
        self.assertEqual(profile.status, "cold")
        self.assertFalse(profile.ready)
        self.assertEqual(idf_weighted_vector("alpha beta", profile), {})

    def test_idf_profile_ready_with_low_test_thresholds(self) -> None:
        docs = ["alpha beta", "beta gamma", "gamma delta"]
        profile = build_idf_profile(docs, min_documents=3, min_unique_terms=4, min_total_tokens=6)
        self.assertEqual(profile.status, "ready")
        self.assertTrue(profile.ready)
        vector = idf_weighted_vector("alpha beta", profile)
        self.assertIn("alpha", vector)
        self.assertIn("beta", vector)
        self.assertGreater(vector["alpha"], vector["beta"])

    def test_idf_profile_roundtrip(self) -> None:
        profile = build_idf_profile(["alpha beta", "beta gamma"], domain="test", min_documents=1, min_unique_terms=1, min_total_tokens=1)
        restored = IdfProfile.from_dict(profile.to_dict())
        self.assertEqual(restored.domain, "test")
        self.assertEqual(restored.status, profile.status)
        self.assertEqual(restored.idf, profile.idf)

    def test_idf_cosine_returns_zero_when_cold(self) -> None:
        profile = build_idf_profile(["alpha beta"], min_documents=10)
        self.assertEqual(idf_cosine_similarity("alpha", "alpha", profile), 0.0)

    def test_idf_cosine_similarity_ready(self) -> None:
        profile = build_idf_profile(
            ["alpha beta", "alpha gamma", "delta epsilon"],
            min_documents=3,
            min_unique_terms=5,
            min_total_tokens=6,
        )
        self.assertGreater(idf_cosine_similarity("alpha beta", "alpha gamma", profile), 0.0)

    def test_domain_profiles_are_independent(self) -> None:
        records = [
            {"domain": "sas", "text": "sas table validation"},
            {"domain": "sas", "text": "sas macro validation"},
            {"domain": "cover_pool", "text": "cover pool collateral"},
        ]
        profiles = build_domain_idf_profiles(records, min_documents=1, min_unique_terms=1, min_total_tokens=1)
        self.assertIn("sas", profiles)
        self.assertIn("cover_pool", profiles)
        self.assertEqual(profiles["sas"].domain, "sas")
