# Concepts

Agent Salience follows a simple flow:

observation -> score -> threshold -> trigger decision -> explanation

## Observation

An observation is any local signal source, typically text or action events provided by a caller.

## Signal

A signal is a measurable representation of an observation. In this package, lexical overlap and simple event repetition become signals.

## Salience

Salience is how strongly one observation stands out relative to a reference or pattern.

## Score

A score is a bounded value in `[0.0, 1.0]` summarizing signal strength.

## Threshold

A threshold is the boundary used to decide whether a score should trigger a response. Thresholds can be fixed, adaptive, or hysteresis-based.

## Trigger

A trigger compares a score to a threshold and returns a structured decision with rationale.

## Resonance

Resonance means a new observation strongly matches a pattern (high score relative to threshold).

## Anti-resonance

Anti-resonance is caller-defined negative-pattern salience. In many integrations it means a strong match against a pattern the caller treats as undesirable.

This package does not hard-code anti-resonance behavior. `kind="anti_resonance"` is metadata for caller logic.

## Drift

Drift is movement away from an anchor signal over time. Here it is modeled as `1 - similarity`.

## Novelty

Novelty is how unlike an observation is compared to known references. High novelty indicates low match with prior examples.

## Loop

A loop is repeated action behavior with low target diversity, such as many calls to the same tool+target pair.

## Explanation

Explanations are short text summaries attached to structured decisions so callers can log, inspect, and tune behavior without opaque heuristics.
