---
type: concept
title: Gradient Cache
slug: gradient-cache
date: 2026-04-20
updated: 2026-04-20
aliases: [梯度缓存, GradCache]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Gradient Cache** (梯度缓存) — a memory-reduction training technique that first computes representation-level gradients for a large contrastive batch, caches them, and then replays encoder backpropagation over smaller sub-batches while preserving the exact full-batch update.

## Key Points

- The paper separates contrastive training into `loss -> representation` and `representation -> encoder parameter`, which makes per-example encoder backpropagation independent once representation gradients are known.
- It performs a graph-less forward pass over the full batch, computes the contrastive loss over stored representations, and caches `u_i = ∂L/∂f(s_i)` and `v_j = ∂L/∂g(t_j)`.
- Encoder gradients are then accumulated by replaying sub-batches with cached representation gradients, yielding the same update as large-batch training instead of the approximation produced by ordinary micro-batching.
- The persistent memory cost is the cached representations or gradients, approximately `(|S|d + |T|d)` floating-point values, rather than full activation tensors for the whole batch.
- In the retrieval experiment, gradient cache enables DPR-style training on a single RTX 2080 Ti with only about `20%` extra runtime from representation pre-computation.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[gao-2021-scaling-2101-06983]]
- [[morris-2024-contextual-2410-02525]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[gao-2021-scaling-2101-06983]].
