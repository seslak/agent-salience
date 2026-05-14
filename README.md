# Agent Salience

Deterministic salience, similarity, threshold, drift, and loop diagnostics for local agentic systems.

Agent Salience is a small **stdlib-only** Python package. It is designed to be embedded by tools such as project memory stores, context-economy tools, governors, and routers. It is not an agent, not an MCP server, not a vector database, and not an embedding model.

The package provides explainable primitives for questions like:

- Is this text similar to that text?
- Is this observation novel or drifting away from the baseline?
- Is this action pattern becoming a loop?
- Did this signal cross a fixed/adaptive/hysteresis threshold?
- Can this local project corpus support IDF-aware scoring yet?
- Can this text be represented by a compact deterministic signature for future indexing?

## Status

Current version: **0.2.0**

Runtime requirements:

- Python **3.9+**
- Standard library only
- No runtime dependencies

## What Agent Salience provides

- Unicode-aware lexical normalization
- Sparse token-frequency maps
- Cosine similarity
- Jaccard similarity with empty/empty as `0.0` by design
- Explainable weighted `signal_score()` breakdowns
- Optional character n-gram similarity for morphological variants
- Optional token-prefix overlap for visible word-family matches
- Optional project-controlled alias-map expansion
- Cold-start-aware local IDF profiles
- Domain-specific IDF profile support
- Deterministic stable hash helper
- Token shingle helpers
- Bounded min-k shingle hashes
- `TextSignature` for future MinHash/LSH preparation
- Welford running statistics
- EWMA statistics
- Adaptive thresholds
- Hysteresis thresholds
- Trigger objects with `evaluate()` / `observe()` behavior
- Drift and novelty scoring
- Repeated-action loop diagnostics
- Compact explanation helpers

## Non-goals

Agent Salience is not:

- an LLM agent
- an MCP server
- a memory store
- a vector database
- an embedding model
- a token counter
- a router
- a governor
- a language-specific stemmer
- a stopword package

It provides deterministic signals. Callers own policy, persistence, routing, enforcement, and domain meaning.

## Installation

From a local checkout:

```bash
pip install .
```

For editable development:

```bash
pip install -e .
```

For no-install local usage, add `src/` to `PYTHONPATH` or `sys.path`:

```python
import sys
from pathlib import Path

ROOT = Path(r"/path/to/agent-salience")
sys.path.insert(0, str(ROOT / "src"))
```

## Quick start

```python
from agent_salience import signal_score

score = signal_score(
    "MCP server entrypoint and tests",
    "Inspect server.py and test_server.py before editing the MCP entrypoint.",
)

print(score.to_dict())
```

`signal_score()` returns an explainable breakdown, not only a scalar:

```python
{
    "cosine": 0.559,
    "jaccard": 0.364,
    "final": 0.500,
    "weights": {"cosine": 0.7, "jaccard": 0.3, ...},
    ...
}
```

Default scoring is deterministic lexical scoring:

```text
final = 0.70 * cosine + 0.30 * jaccard
```

Optional components are explicit and off by default unless requested through weights/options.

## Similarity basics

### Jaccard

```python
from agent_salience import jaccard_similarity

jaccard_similarity(["agent", "memory"], ["agent", "routing"])
```

`jaccard_similarity([], [])` returns `0.0` because, in search and memory salience, empty text is treated as **no evidence**, not as a perfect semantic match.

### Cosine

```python
from agent_salience import cosine_similarity, token_frequencies

left = token_frequencies("agent loop budget")
right = token_frequencies("agent budget discipline")

cosine_similarity(left, right)
```

## Optional fuzzy lexical helpers

### Character n-grams

Character n-gram similarity helps with visible morphological variants:

```python
from agent_salience import char_ngram_similarity

char_ngram_similarity("validation", "validated")
char_ngram_similarity("configuration", "config")
```

This is language-agnostic. It does not solve conceptual synonyms.

### Token-prefix overlap

Token-prefix overlap gives a small similarity signal when tokens share a visible prefix:

```python
from agent_salience import normalize_text, token_prefix_overlap

left = normalize_text("configuration")
right = normalize_text("config")

token_prefix_overlap(left, right)
```

This is not stemming and not alias expansion. It is a deterministic word-family helper.

## Project-controlled alias maps

Alias maps bridge local project vocabulary without using embeddings or hardcoded language stopwords.

```python
from agent_salience import signal_score

aliases = {
    "test_failure": [
        "test failure",
        "broken validation",
        "debug failing test",
        "failed validation run",
    ],
}

score = signal_score(
    "test failure triage",
    "debug broken validation run",
    alias_map=aliases,
    weights={"cosine": 0.4, "jaccard": 0.3, "alias": 0.3},
)
```

Alias governance belongs to the caller, usually a coordinator/router policy layer. Agent Salience only applies the approved alias map it receives.

## IDF support

IDF support is local, language-agnostic, and cold-start aware.

```python
from agent_salience import build_idf_profile, signal_score

corpus = [
    "mnemo stores project memory",
    "thrift tracks token economy",
    "governor detects repeated loops",
]

profile = build_idf_profile(
    corpus,
    min_documents=3,
    min_unique_terms=5,
    min_total_tokens=6,
)

score = signal_score(
    "token economy",
    "context cost tracking",
    mode="auto",
    idf_profile=profile,
    weights={"cosine": 0.5, "jaccard": 0.2, "idf_cosine": 0.3},
)

print(score.idf_status, score.idf_used)
```

When the corpus is too small, IDF stays cold:

```text
idf_status = "cold"
idf_used = false
```

In cold mode, scoring falls back to lexical components. IDF does not replace Jaccard; it adds an optional weighted component when enough local corpus exists.

### Domain-specific IDF

```python
from agent_salience import build_domain_idf_profiles

records = [
    {"domain": "memory", "text": "mnemo recall context block"},
    {"domain": "memory", "text": "hippocampus durable entry"},
    {"domain": "economy", "text": "token budget context window"},
]

profiles = build_domain_idf_profiles(records)
```

Domain IDF is useful when different areas develop different common vocabulary.

## Text signatures and future LSH preparation

Agent Salience `0.2.0` does **not** implement full MinHash/LSH.

It does provide stable signature primitives so callers can prepare storage schemas now and add more advanced indexing later:

```python
from agent_salience import build_text_signature

signature = build_text_signature("Run tests before release handoff.")
print(signature.to_dict())
```

A signature includes:

- `content_hash`
- `normalized_hash`
- `token_count`
- `unique_token_count`
- `top_terms`
- `shingle_hashes`
- `signature_version`
- `normalizer_version`

This makes future MinHash/LSH a signature-version upgrade instead of a redesign.

## Loop diagnostics

```python
from agent_salience import ActionEvent, detect_repeated_target_loop

events = [
    ActionEvent(tool="file_window", target="server.py"),
    ActionEvent(tool="file_window", target="server.py"),
    ActionEvent(tool="file_window", target="server.py"),
    ActionEvent(tool="grep_text", target="server.py"),
]

decision = detect_repeated_target_loop(events, threshold=0.6, min_count=3)
print(decision.to_dict())
```

Loop decisions are diagnostics. Callers decide whether to warn, pause, stop, or ignore.

## Drift and novelty

```python
from agent_salience import drift_score, novelty_score

baseline = "Fix Mnemo consolidation and signature backfill."
current = "Discuss apartment prices and kindergarten logistics."

drift = drift_score(baseline, current)
```

Drift is a mission-alignment signal. It should feed policy; it should not be treated as an automatic stop button by itself.

## Thresholds and triggers

```python
from agent_salience import AdaptiveThreshold, SignalTrigger

threshold = AdaptiveThreshold(base=0.70, minimum=0.45, maximum=0.95)
trigger = SignalTrigger(
    name="loop-warning",
    pattern="repeated same tool and target",
    threshold=threshold,
    kind="loop",
)

decision = trigger.observe("same file_window target repeated repeatedly")
print(decision.to_dict())
```

Thresholds support serialization/roundtrip so callers can persist local adaptive state.

## Integration model

Agent Salience is meant to be embedded by other local tools:

| Caller | Typical use |
|---|---|
| Project memory | relevance, duplicate candidates, novelty, signature helpers |
| Context economy tool | repeated file-window/read patterns, cost-related similarity |
| Governor | loop/no-progress diagnostics, drift warnings |
| Router | task similarity, known-workflow matching, alias/IDF-assisted routing |
| Feedback writer | deciding which lessons are worth recording or promoting |

Separation of responsibilities:

```text
Agent Salience = deterministic signal primitives
Caller = storage, policy, routing, enforcement, approvals
```

## Documentation

- [Concepts](docs/concepts.md)
- [Formulas](docs/formulas.md)
- [Integration](docs/integration.md)
- [Similarity Extensions](docs/similarity_extensions.md)

## Development

Run from repository root:

```bash
python -m compileall -q .
python smoke_test.py
python -m unittest discover -s tests -p "test*.py"
```

Expected result for `0.2.0`:

```text
57 tests passed
```

## Stability notes

`0.2.0` keeps the original lexical default behavior and adds optional extension points. Full LSH, embeddings, persistent corpus stores, and caller-specific policies are intentionally out of scope.

## License

MIT. See [LICENSE](LICENSE).
