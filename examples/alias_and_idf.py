"""Alias-map and IDF usage example."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agent_salience import build_idf_profile, signal_score

aliases = {
    "test_failure": [
        "test failure",
        "broken validation",
        "debug failing test",
        "failed validation run",
    ]
}

profile = build_idf_profile(
    [
        "test failure triage",
        "debug broken validation run",
        "fix failed validation test",
        "memory gateway record search recall",
    ],
    min_documents=4,
    min_unique_terms=5,
    min_total_tokens=8,
)

score = signal_score(
    "test failure triage",
    "debug broken validation run",
    alias_map=aliases,
    idf_profile=profile,
    mode="auto",
    weights={"cosine": 0.40, "jaccard": 0.20, "alias": 0.20, "idf_cosine": 0.20},
)

print(score.to_dict())
