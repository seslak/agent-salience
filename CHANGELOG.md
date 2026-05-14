# Changelog

## 0.2.0

- Added deterministic signature preparation helpers: stable hashes, token shingles, bounded shingle hashes, and `TextSignature`.
- Added language-agnostic fuzzy lexical helpers: character n-gram similarity and token-prefix overlap.
- Added project-controlled alias-map expansion for local synonym/vocabulary bridges.
- Added cold-start-aware local IDF profiles, including domain-specific profile generation.
- Extended `signal_score()` with optional fuzzy, alias, and IDF components while keeping default lexical behavior unchanged.
- Documented future LSH and IDF paths without enabling approximate LSH behavior by default.

## 0.1.1

- Added Unicode-aware tokenization that preserves non-ASCII letters and splits underscores.
- Replaced Python 3.10 union type syntax with Python 3.9-compatible `Optional`/`Union` hints.
- Changed default signal weights to lexical-only scoring (`cosine`/`jaccard`) so exact text matches score near 1.0.
- Clarified `anti_resonance` as caller-defined negative-pattern salience, not hard-coded behavior.
- Added `to_dict()` serialization helpers on scoring and decision dataclasses, plus `ActionEvent.from_dict()`.
- Added a package smoke test for version, Unicode normalization, scoring, trigger evaluation, and loop detection.

## 0.1.0

- Initial release of `agent-salience`.
- Added deterministic lexical scoring primitives.
- Added adaptive threshold and hysteresis decisions.
- Added drift, novelty, and repeated-action loop diagnostics.
- Added explanation helpers for compact decision reporting.
