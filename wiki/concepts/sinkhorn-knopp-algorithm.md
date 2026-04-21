---
type: concept
title: Sinkhorn-Knopp Algorithm
slug: sinkhorn-knopp-algorithm
date: 2026-04-20
updated: 2026-04-20
aliases: [Sinkhorn normalization, Sinkhorn-Knopp, Sinkhorn-Knopp 算法]
tags: []
source_count: 1
confidence: low
domain_volatility: low
last_reviewed: 2026-04-20
---

## Definition

**Sinkhorn-Knopp Algorithm** (Sinkhorn-Knopp 算法) — an iterative normalization procedure that alternates row and column rescaling to turn a positive matrix into an approximately doubly stochastic one.

## Key Points

- mHC uses Sinkhorn-Knopp to project the unconstrained residual mapping `H~_l^res` onto the doubly stochastic manifold.
- The procedure starts from `M^(0) = exp(H~_l^res)` and iterates `M^(t) = T_r(T_c(M^(t-1)))` until the row and column sums approach `1`.
- The paper uses `t_max = 20` iterations as a practical compromise between projection quality and runtime efficiency.
- A custom backward kernel recomputes the intermediate iteration states on-chip instead of storing all Sinkhorn activations.
- Because the iteration is truncated, the resulting mapping is only approximately doubly stochastic, which explains the small residual deviation from ideal unit gain in the stability plots.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[xie-2026-mhc-2512-24880]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[xie-2026-mhc-2512-24880]].
