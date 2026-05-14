# Release Checklist

Use this checklist before tagging an Agent Salience release.

## Validation

Run from repository root:

```bash
python -m compileall -q .
python smoke_test.py
python -m unittest discover -s tests -p "test*.py"
```

## Metadata

- [ ] `src/agent_salience/__init__.py` version is updated
- [ ] `pyproject.toml` version is updated
- [ ] `CHANGELOG.md` has an entry for the release
- [ ] README reflects public API changes
- [ ] Docs reflect public API changes

## Repository hygiene

- [ ] No `__pycache__/`
- [ ] No `*.pyc`
- [ ] No local `.venv/`
- [ ] No generated logs or scratch files

## Tagging

```bash
git status
git add .
git commit -m "Release Agent Salience vX.Y.Z"
git tag vX.Y.Z
git push
git push origin vX.Y.Z
```
