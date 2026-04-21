---
type: concept
title: MaxSim Scoring
slug: maxsim-scoring
date: 2026-04-20
updated: 2026-04-20
aliases: [maximum-similarity scoring]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**MaxSim Scoring** — a token-level relevance computation that, for each query token, keeps only the maximum similarity over document tokens and sums those maxima into the final score.

## Key Points

- PLAID uses MaxSim twice: first approximately over centroid scores, and then exactly after reconstructing residual-compressed passage embeddings.
- The late-interaction score is `S_{q,d} = \sum_i \max_j Q_i \cdot D_j^T`, which makes passage length and padding strategy important for efficient implementation.
- Vanilla ColBERTv2 relies on padded tensors for MaxSim, whereas PLAID introduces padding-free CPU kernels over packed 2D tensors.
- The paper identifies MaxSim execution and the surrounding data movement as one of the dominant contributors to end-to-end retrieval latency.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[santhanam-2022-plaid-2205-09707]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[santhanam-2022-plaid-2205-09707]].
