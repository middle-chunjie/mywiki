---
type: concept
title: Weight Sharing
slug: weight-sharing
date: 2026-04-20
updated: 2026-04-20
aliases: [shared weights, 权重共享]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Weight Sharing** (权重共享) — a design strategy in which multiple model components reuse the same trainable parameters to induce alignment, regularization, or efficiency.

## Key Points

- The cross-modal adapter shares part of the adapter up-projection matrix between the video and text encoders.
- The shared width `d_s` determines how much of the adapter output is common across modalities and is tuned per dataset.
- Sharing weights gives consistent but modest retrieval improvements over the non-shared adapter baseline.
- The paper interprets weight sharing as a way to re-align CLIP's vision and language feature spaces without explicit feature fusion.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[jiang-2022-crossmodal-2211-09623]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[jiang-2022-crossmodal-2211-09623]].
