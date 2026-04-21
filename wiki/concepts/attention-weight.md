---
type: concept
title: Attention Weight
slug: attention-weight
date: 2026-04-20
updated: 2026-04-20
aliases: [attention distribution, 注意力权重]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Attention Weight** (注意力权重) — the normalized coefficient assigned by an attention mechanism to quantify how much one token should influence another.

## Key Points

- This paper studies attention weights from the target position to demonstration label words as a proxy for class preference.
- Deep-layer attention to label anchors reaches high predictive correlation with final outputs, with `AUCROC` near `0.8`.
- The proposed anchor re-weighting method explicitly rescales selected attention weights by `exp(β_0^i)`.
- The work treats attention weights as useful but incomplete evidence, complementing them with intervention and representation-distance analyses.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[wang-2023-label]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[wang-2023-label]].
