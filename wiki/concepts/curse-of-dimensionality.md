---
type: concept
title: Curse of Dimensionality
slug: curse-of-dimensionality
date: 2026-04-20
updated: 2026-04-20
aliases: [dimension curse, 维度灾难]
tags: []
source_count: 1
confidence: low
domain_volatility: low
last_reviewed: 2026-04-20
---

## Definition

**Curse of Dimensionality** (维度灾难) — the phenomenon that approximation, optimization, or sample complexity grows rapidly with input dimension for general high-dimensional function classes.

## Key Points

- The paper motivates KANs partly as a way to avoid the curse of dimensionality when a target function admits a smooth compositional decomposition into univariate functions and sums.
- Its approximation theorem argues that spline error can scale with grid resolution `G` rather than directly with ambient dimension, giving rates like `G^{-k-1+m}` under strong assumptions.
- Toy and special-function experiments are presented as empirical evidence that KANs exploit low-dimensional internal structure better than MLPs.
- The paper explicitly notes that this escape from the curse is not universal: the constant in the bound still depends on the underlying representation, and arbitrary functions remain hard.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[liu-2024-kan-2404-19756]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[liu-2024-kan-2404-19756]].
