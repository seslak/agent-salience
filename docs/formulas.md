# Formulas

## Cosine similarity

For sparse maps `a` and `b`:

```text
cos(a, b) = dot(a, b) / (||a|| * ||b||)
```

where:

```text
dot(a, b) = sum(a_i * b_i)
||a|| = sqrt(sum(a_i^2))
```

Output is clamped to `[0.0, 1.0]`.

## Jaccard similarity

For token sets `A` and `B`:

```text
J(A, B) = |A intersect B| / |A union B|
```

Output is clamped to `[0.0, 1.0]`, with `0.0` for two empty sets.

## Weighted signal score

Let each bounded component be in `[0, 1]`:

```text
S = w_cos * cosine
  + w_jac * jaccard
  + w_rep * repetition
  + w_rec * recency
  + w_nov * novelty
  + w_drf * drift
```

Weights are normalized to sum to `1.0` when possible.

## Welford running stats

For each new value `x`:

```text
delta = x - mean
mean_new = mean + delta / n
m2_new = m2 + delta * (x - mean_new)
```

Sample variance:

```text
variance = m2 / (n - 1), for n > 1
```

## EWMA mean

```text
mu_t = alpha * x_t + (1 - alpha) * mu_{t-1}
```

with `alpha in (0, 1]`.

## EWMA variance (approximate)

```text
var_t = alpha * (x_t - mu_t)^2 + (1 - alpha) * var_{t-1}
```

## Adaptive threshold

When stats are available:

```text
threshold = clamp(mean + k * stddev, minimum, maximum)
```

Fallback when stats are not initialized:

```text
threshold = clamp(base, minimum, maximum)
```

## Hysteresis threshold

- inactive -> active when `value >= enter`
- active -> inactive when `value < exit`

Typically `exit <= enter` to prevent rapid state flapping.

## Drift

```text
drift = 1 - signal_score(anchor, new).final
```

In this package defaults, `signal_score` is lexical-only (`cosine`, `jaccard`) unless callers provide additional non-zero weights.

## Novelty

```text
novelty = 1 - max(signal_score(text, ref).final for ref in references)
```

If no references exist: `novelty = 1.0`.

## Repetition score

Given event list `E`:

```text
repetition_score = matching_events / total_events
```

Result is clamped to `[0.0, 1.0]`.
