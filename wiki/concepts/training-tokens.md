---
type: concept
title: Training Tokens
slug: training-tokens
date: 2026-04-20
updated: 2026-04-20
aliases: [token budget, 训练 token 数]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Training Tokens** (训练 token 数) — the total number of tokens consumed during model training, treated as a primary scaling variable alongside model size and compute.

## Key Points

- This paper argues that the token budget has been systematically under-scaled in large language model training.
- For the Gopher compute budget, the predicted compute-optimal token count is around `1.4T` to `1.7T`, far above the `300B` tokens used by many contemporary models.
- The paper reports that a `175B` model would require on the order of `3.7T` to `4.3T` tokens to lie on the estimated efficient frontier.
- Matching the cosine learning-rate schedule to the intended token horizon is necessary for token-budget comparisons to be meaningful.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[hoffmann-2022-training-2203-15556]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[hoffmann-2022-training-2203-15556]].
