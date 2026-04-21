---
type: concept
title: Birkhoff Polytope
slug: birkhoff-polytope
date: 2026-04-20
updated: 2026-04-20
aliases: [Birkhoff-von Neumann polytope, Birkhoff 多面体]
tags: []
source_count: 1
confidence: low
domain_volatility: low
last_reviewed: 2026-04-20
---

## Definition

**Birkhoff Polytope** (Birkhoff 多面体) — the convex polytope whose points are doubly stochastic matrices, equivalently the convex hull of permutation matrices.

## Key Points

- The paper identifies the feasible set for `H_l^res` as the Birkhoff polytope and uses it as the geometric target of mHC projection.
- This interpretation lets each residual update act as a convex combination of stream permutations rather than as an arbitrary dense linear map.
- The geometry matters because it preserves cross-stream mixing while preventing the unbounded gain behavior seen in unconstrained HC.
- The authors use the polytope view to argue that repeated application of `H_l^res` produces robust feature fusion instead of unstable amplification.
- For `n = 1`, the polytope collapses to the trivial identity case, linking the generalized setting back to ordinary residual connections.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[xie-2026-mhc-2512-24880]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[xie-2026-mhc-2512-24880]].
