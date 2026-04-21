---
type: concept
title: Budget Allocation
slug: budget-allocation
date: 2026-04-20
updated: 2026-04-20
aliases: [Resource Allocation, 预算分配]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Budget Allocation** (预算分配) — the problem of choosing how to distribute limited resources across data collection, model training, and deployment costs to optimize system performance.

## Key Points

- The paper combines model-size and data-size scaling into a joint fit, `` `L(N, D) = [ (A / N)^(alpha / beta) + B / D ]^beta + delta` ``, to predict retrieval quality under different resource choices.
- It defines total cost as `` `Z(N, D) = Z_data · D + Z_train · N + Z_infer · N` ``, separating annotation, training, and inference costs.
- The estimated annotation cost is `0.6` dollars per query-passage pair, while the fitted compute surrogates are `` `Z_train ≈ 3.22e-8` `` and `` `Z_infer ≈ 0.43` ``.
- Without inference cost, the predicted optimum shifts toward very large models and can exceed `13B` parameters.
- Once corpus-wide encoding cost is included, the optimum drops sharply to million-scale retrievers, highlighting a key difference between retrieval and autoregressive generation.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[fang-2024-scaling-2403-18684]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[fang-2024-scaling-2403-18684]].
