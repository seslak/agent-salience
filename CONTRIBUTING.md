# Contributing to Agent Salience

Thanks for considering a contribution.

Agent Salience is intentionally small, deterministic, and stdlib-only. Contributions should preserve that shape unless there is a very strong reason to change it.

## Design principles

- Keep runtime dependencies at zero.
- Keep default scoring deterministic and explainable.
- Keep caller policy outside this package.
- Prefer explicit optional components over hidden behavior changes.
- Treat empty/tiny text as no evidence unless a higher-level caller policy says otherwise.
- Do not add language-specific stopword/stemmer behavior to the default path.
- Do not introduce embeddings or approximate indexes into core without clear opt-in boundaries.

## Local validation

Run from repository root:

```bash
python -m compileall -q .
python smoke_test.py
python -m unittest discover -s tests -p "test*.py"
```

## Python compatibility

The package currently targets Python 3.9+ and uses only the standard library.

Avoid syntax that breaks Python 3.9, including PEP 604 union syntax such as `str | None`.

## Tests

Add or update tests when changing:

- token normalization
- similarity semantics
- threshold behavior
- loop detection
- IDF readiness thresholds
- alias expansion
- signature helpers
- public dataclass serialization

Tests should prefer exact expected values for stable contracts and invariant-based checks for edge cases where the exact implementation detail is not intended as a public contract.

## Documentation

Update the README and relevant files under `docs/` when adding public functionality.

## Pull request checklist

Before submitting:

- [ ] `python -m compileall -q .` passes
- [ ] `python smoke_test.py` passes
- [ ] `python -m unittest discover -s tests -p "test*.py"` passes
- [ ] README/docs updated if public API changed
- [ ] No generated artifacts or caches committed
