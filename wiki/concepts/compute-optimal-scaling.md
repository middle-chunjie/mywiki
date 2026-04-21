---
type: concept
title: Compute-Optimal Scaling
slug: compute-optimal-scaling
date: 2026-04-20
updated: 2026-04-20
aliases: [compute optimal training, 计算最优扩展]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Compute-Optimal Scaling** (计算最优扩展) — the problem of choosing model size and training duration so that a fixed compute budget yields the lowest attainable final loss or best downstream performance.

## Key Points

- The paper defines compute-optimality as minimizing `L(N,D)` subject to the constraint `FLOPs(N,D) = C`.
- All three fitted approaches imply that many frontier dense LLMs were over-parameterized and under-trained for their budgets.
- The central practical prescription is to scale parameters and training tokens in roughly equal proportion as compute grows.
- Chinchilla is presented as an empirical validation of this rule at the Gopher compute budget.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[hoffmann-2022-training-2203-15556]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[hoffmann-2022-training-2203-15556]].
