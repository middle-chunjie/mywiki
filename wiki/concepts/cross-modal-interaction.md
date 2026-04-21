---
type: concept
title: Cross-Modal Interaction
slug: cross-modal-interaction
date: 2026-04-20
updated: 2026-04-20
aliases: [cross-modal communication, 跨模态交互]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Cross-Modal Interaction** (跨模态交互) — information exchange between representations from different modalities so that their features can be aligned or fused for a downstream task.

## Key Points

- The paper argues that independent uni-modal adapters are sub-optimal because they block early interactions between video and text representations.
- Direct explicit interaction inside shallow layers would require pairwise text-video coupling and raise complexity from `O(N)` to `O(N^2)` over batch pairs.
- The proposed cross-modal adapter implements implicit early interaction by sharing part of the adapter up-projection across modalities.
- Ablation shows the interaction mechanism improves retrieval `R@1` on all five datasets compared with vanilla adapters.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[jiang-2022-crossmodal-2211-09623]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[jiang-2022-crossmodal-2211-09623]].
