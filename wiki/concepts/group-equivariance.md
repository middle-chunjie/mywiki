---
type: concept
title: Group Equivariance
slug: group-equivariance
date: 2026-04-20
updated: 2026-04-20
aliases: [group equivariance, 群等变性]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Group Equivariance** (群等变性) — the property that applying a group transformation before a function yields the same result as applying the corresponding transformation after the function.

## Key Points

- The paper requires the representation learner `r` to satisfy `r(g(c)) = g(r(c))` for all `g` in the target code-symmetry group.
- Equivariance is preferred over invariant intermediate representations because it preserves information about how code was transformed.
- Embedding layers and the graph-aware attention layers are both proven equivariant under the chosen automorphism group.
- The paper's ablation shows that forcing invariance too early degrades performance by up to `60.7%` relative to the fully equivariant design.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[pei-2024-exploiting-2308-03312]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[pei-2024-exploiting-2308-03312]].
