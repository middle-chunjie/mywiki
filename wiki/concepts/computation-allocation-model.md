---
type: concept
title: Computation Allocation Model
slug: computation-allocation-model
date: 2026-04-20
updated: 2026-04-20
aliases: [computation allocation model, compute allocation model]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Computation allocation model** — a predictive model that estimates task performance from how inference compute is distributed across retrieval, demonstrations, and iterative generation parameters.

## Key Points

- The paper models RAG performance as a function of `theta = (k, m, n)` and task-specific informativeness terms.
- It applies an inverse-sigmoid transformation to performance before fitting a log-linear relation with ordinary least squares.
- The fitted model is used to predict near-optimal inference configurations under a fixed effective-context budget.
- Empirically, the model generalizes to unseen domains and longer context budgets better than simple baseline allocations.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[unknown-nd-inference]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[unknown-nd-inference]].
