---
type: concept
title: Unbiased Pass@k
slug: unbiased-pass-at-k
date: 2026-04-20
updated: 2026-04-20
aliases: [unbiased Pass@k, pass@k estimator, 无偏 Pass@k]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Unbiased Pass@k** (无偏 Pass@k) — an estimator of code-generation success that corrects for sampling bias by computing the probability that at least one of `k` sampled programs is correct given `n` total samples and `c` correct ones.

## Key Points

- The paper uses unbiased Pass@k instead of naive top-k counting because prior work found the naive estimate has high variance.
- Their experiments fix `n = 20` sampled programs per requirement and report results for `k ∈ {1, 3, 5}`.
- The estimator is written as `1 - C(n - c, k) / C(n, k)` averaged over problems.
- This metric lets the authors compare prompting strategies under the same sampling budget rather than a single deterministic output.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[li-2023-structured-2305-06599]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[li-2023-structured-2305-06599]].
