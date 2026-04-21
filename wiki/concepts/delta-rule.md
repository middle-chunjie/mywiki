---
type: concept
title: Delta rule
slug: delta-rule
date: 2026-04-20
updated: 2026-04-20
aliases: [Widrow-Hoff rule]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Delta rule** — an online update rule that adjusts a memory or weight matrix toward reconstructing the target value associated with the current key.

## Key Points

- The paper interprets linear attention as online learning over a matrix-valued state `S_t` and uses the delta rule to turn key-value storage into a reconstruction objective.
- Kimi Linear inherits the gated-delta lineage from DeltaNet and Gated DeltaNet, but replaces coarse scalar forgetting with channel-wise decay.
- In KDA, the update `(I - β_t k_t k_t^T) Diag(α_t) S_{t-1} + β_t k_t v_t^T` preserves the rank-1 structure needed for efficient chunkwise parallelization.
- The authors argue that delta-rule-based updates improve memory correction and expressivity relative to plain multiplicative-decay linear attention.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[team-2025-kimi-2510-26692]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[team-2025-kimi-2510-26692]].
