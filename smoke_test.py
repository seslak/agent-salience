#!/usr/bin/env python3
"""Smoke test for agent-salience."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import agent_salience
from agent_salience import (
    ActionEvent,
    AdaptiveThreshold,
    SignalTrigger,
    detect_repeated_target_loop,
    normalize_text,
    signal_score,
)


def main() -> int:
    assert agent_salience.__version__ == "0.2.0"

    tokens = normalize_text("Šešlak čuva žutu Đurđevdan Überstraße groß")
    assert tokens == ["šešlak", "čuva", "žutu", "đurđevdan", "überstraße", "groß"]

    identical = signal_score("alpha beta", "alpha beta")
    assert identical.final >= 0.999

    trigger = SignalTrigger(
        name="smoke-trigger",
        pattern="alpha beta",
        threshold=AdaptiveThreshold(base=0.7, minimum=0.4, maximum=0.95),
        kind="resonance",
    )
    decision = trigger.evaluate("alpha beta")
    assert decision.triggered

    events = [ActionEvent(tool="file_window", target="README.md") for _ in range(6)]
    loop = detect_repeated_target_loop(events, threshold=0.6, min_count=5)
    assert loop.triggered

    print("OK: agent-salience smoke test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
