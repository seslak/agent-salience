# Similarity Extensions

Agent Salience keeps a small deterministic lexical baseline and adds optional extension points. These extensions are explicit because they change score interpretation.

## Baseline

The baseline is lexical:

- normalized token cosine
- token-set Jaccard

Empty/tiny texts are treated as no evidence. `jaccard_similarity([], [])` returns `0.0`.

## Morphological variants

Use character n-grams and token-prefix overlap for visible word-family matches:

```python
from agent_salience import char_ngram_similarity, token_prefix_overlap, normalize_text

char_ngram_similarity("validation", "validated")
token_prefix_overlap(normalize_text("configuration"), normalize_text("config"))
```

These helpers are language-agnostic. They do not solve conceptual synonyms.

## Alias maps

Alias maps bridge project-specific vocabulary:

```python
aliases = {
    "test_failure": [
        "test failure",
        "broken validation",
        "debug failing test",
    ]
}
```

Agent Salience can apply an approved alias map, but it does not own alias policy. Alias maps should be maintained by the coordinator/router policy layer, with candidate updates written by a cheap feedback/vocabulary role and approved before becoming active.

## IDF

IDF is useful only after enough local corpus exists.

Agent Salience supports cold-start-aware IDF profiles:

```python
from agent_salience import build_idf_profile

profile = build_idf_profile(documents)
```

If the profile is cold, IDF vectors are empty and `signal_score(..., mode="auto")` falls back to lexical scoring while reporting `idf_status="cold"` and `idf_used=False`.

Domain-specific profiles are supported:

```python
from agent_salience import build_domain_idf_profiles

profiles = build_domain_idf_profiles(records)
```

Start with project-level IDF and use domain-specific IDF when a domain has enough local context.

## LSH preparation

Agent Salience 0.2.0 does not implement LSH. It provides stable signatures:

- `stable_hash_hex()`
- `token_shingles()`
- `shingle_hashes()`
- `TextSignature`
- `build_text_signature()`

These make later MinHash/LSH possible without changing caller contracts.
