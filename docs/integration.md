# Integration

`agent-salience` is a library of deterministic primitives. Domain behavior belongs to the caller.

## Integration Rules

- Keep persistence in the caller.
- Keep domain-specific meanings in the caller.
- Use structured decisions (`SignalDecision`, `ThresholdDecision`, `LoopDecision`).
- Start with warnings and diagnostics first.
- Do not silently block actions at first.
- Let the caller decide whether a trigger matters.

## Local No-Install Usage

A caller can load this package directly from source:

```python
import os
import sys

root = os.environ.get("AGENT_SALIENCE_HOME", "/path/to/agent-salience")
sys.path.insert(0, os.path.join(root, "src"))
```

You can also add `agent-salience/src` directly to `sys.path` in tests or tooling.

## Example Patterns

## Memory duplicate detection

- Caller computes `signal_score(existing_item, new_item)`.
- Caller compares against an adaptive threshold.
- Caller emits a warning when likely duplicate; no auto-merge yet.

## Invariant drift warning

- Store baseline invariant text in caller state.
- Compute `drift_score(baseline, current_text)`.
- Warn when drift exceeds caller-defined limits.

## Repeated file-window loop warning

- Convert tool calls to `ActionEvent`.
- Run `detect_repeated_target_loop`.
- Surface warning when repeated reads dominate.

## Context economy warning

- Track recent action events and estimated costs in caller state.
- Use repetition score plus salience drift to flag wasteful patterns.

## Anti-pattern detection

- Define pattern text for known anti-patterns.
- Run `SignalTrigger(kind="anti_resonance")` or caller-specific labels.
- Treat anti-resonance as caller-defined negative-pattern salience, usually high match against undesirable text.
- Keep policy decisions in the caller; this library does not enforce behavior by `kind`.
- Log explainable decisions for tuning before enforcing policy.
