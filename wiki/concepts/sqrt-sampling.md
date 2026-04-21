---
type: concept
title: SQRT Sampling
slug: sqrt-sampling
date: 2026-04-20
updated: 2026-04-20
aliases: [SQRT Sampling, sqrt sampling]
tags: [search, sampling]
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**SQRT Sampling** — a sampling strategy that draws candidates from the normalized square root of a target distribution to minimize expected search loss among memoryless samplers when the square-root mass is finite.

## Key Points

- [[fijalkow-2022-scaling]] proves that `sqrt(D)(x) = sqrt(D(x)) / sum_y sqrt(D(y))` is loss optimal among all sampling algorithms whenever `sum_x sqrt(D(x)) < infinity`.
- The method is memoryless, so unlike enumerative methods it may resample the same program multiple times but requires very little state.
- For PCFGs, the paper shows how to implement SQRT Sampling by square-rooting rule probabilities and then renormalizing the grammar.
- In experiments, SQRT Sampling attains `14,020` programs/s and scales nearly linearly under grammar splitting, with `2.8x` more programs when moving from `2` to `6` CPUs.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[fijalkow-2022-scaling]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[fijalkow-2022-scaling]].
