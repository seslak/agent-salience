"""Basic Agent Salience usage example."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agent_salience import signal_score

score = signal_score(
    "MCP server entrypoint and tests",
    "Inspect server.py and test_server.py before editing the MCP entrypoint.",
)

print(score.to_dict())
