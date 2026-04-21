---
type: concept
title: Soft Nearest Neighbor Loss
slug: soft-nearest-neighbor-loss
date: 2026-04-20
updated: 2026-04-20
aliases: [SNN loss, soft nearest neighbor loss, 软最近邻损失]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Soft Nearest Neighbor Loss** (软最近邻损失) — a metric-learning objective that maximizes the probability that a sample's stochastic neighbor comes from the same semantic group rather than from other groups in the batch.

## Key Points

- Syntriever adapts soft nearest neighbor loss to retrieval by treating `(q_i, p_i, p_i^+, q_i^cot)` as an entangled positive group rather than a standard classification class.
- The numerator contains attraction terms for the labeled passage, the synthetic positive passage, and the chain-of-thought query expansion.
- Synthetic hard negatives and in-batch negatives appear only in the denominator, so they contribute repulsion without being pulled toward the query.
- The paper uses this loss to preserve multiple relevant views of one query instead of collapsing training to a single positive pair.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[kim-2025-syntriever-2502-03824]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[kim-2025-syntriever-2502-03824]].
