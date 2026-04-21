---
type: concept
title: Pruning Mask
slug: pruning-mask
date: 2026-04-20
updated: 2026-04-20
aliases: [pruning masks, mask variable, 剪枝掩码]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Pruning Mask** (剪枝掩码) — a learned gating variable that controls whether a parameter group or architectural substructure is retained or removed during pruning.

## Key Points

- The paper defines separate mask families for layers, hidden dimensions, attention heads, and FFN intermediate dimensions.
- Learned masks let the method search over subnetworks while still training model weights jointly with the pruning decisions.
- After optimization, the authors finalize architecture by keeping the highest-scoring components associated with each mask variable.
- Mask design is central to expressing target shape constraints in a way that yields dense deployable models rather than irregular sparsity patterns.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[xia-2024-sheared-2310-06694]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[xia-2024-sheared-2310-06694]].
