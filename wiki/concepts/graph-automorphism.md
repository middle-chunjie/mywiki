---
type: concept
title: Graph Automorphism
slug: graph-automorphism
date: 2026-04-20
updated: 2026-04-20
aliases: [graph automorphism, 图自同构]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Graph Automorphism** (图自同构) — a bijection on a graph's nodes that preserves adjacency structure and therefore leaves the graph unchanged up to node relabeling.

## Key Points

- The paper models semantics-preserving code permutations through automorphisms of the program interpretation graph.
- `Aut(IG)` is used as the key symmetry group because preserving graph connectivity is argued to preserve program input-output behavior.
- The modified attention bias must stay invariant under these automorphisms and commute with the associated permutation matrices.
- In practice, the implementation works with `Aut(PDG)` rather than direct `Aut(IG)` because PDG can be built statically.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[pei-2024-exploiting-2308-03312]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[pei-2024-exploiting-2308-03312]].
