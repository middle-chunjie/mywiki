---
type: concept
title: Cross-Modal Alignment
slug: cross-modal-alignment
date: 2026-04-20
updated: 2026-04-20
aliases: [跨模态对齐]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Cross-Modal Alignment** (跨模态对齐) — the process of mapping representations from different modalities into a shared embedding space where semantically matched items are close.

## Key Points

- This paper aligns point-cloud, image, and text representations in a shared CLIP-compatible space for open-world 3D understanding.
- The baseline uses paired point-image and point-text losses, while MixCon3D adds an extra image-plus-point-cloud joint representation aligned to text.
- The authors keep the image-text CLIP loss during training so the frozen 2D and text spaces remain explicitly connected to the learned 3D branch.
- Better cross-modal alignment is reflected in both zero-shot recognition gains and stronger qualitative text-to-3D retrieval and captioning behavior.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[gao-2024-sculpting-2311-01734]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[gao-2024-sculpting-2311-01734]].
