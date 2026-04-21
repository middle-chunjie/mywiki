---
type: concept
title: Layer-Wise Expansion
slug: layer-wise-expansion
date: 2026-04-20
updated: 2026-04-20
aliases: [layerwise expansion, hierarchical expansion, 逐层扩展]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Layer-Wise Expansion** (逐层扩展) — a question-expansion strategy that traverses the leaves of a formal reasoning graph and replaces one leaf constant at a time with a deeper sub-question, preserving the original answer while increasing structural depth.

## Key Points

- WebShaper applies expansion only to leaf constants in the KP graph instead of randomly attaching new facts anywhere.
- The strategy is designed to avoid redundant constant-to-constant facts that do not lengthen the reasoning path.
- It also blocks reasoning shortcuts where a newly added fact connects too directly to the final target variable.
- Each expansion step keeps the answer invariant through `q^{n+1}(T) = \mathrm{Expander}(C, q^n(T))`.
- Ablation experiments show layer-wise expansion outperforms a sequential-structure baseline on all tested base models.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[tao-2025-webshaper-2507-15061]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[tao-2025-webshaper-2507-15061]].
