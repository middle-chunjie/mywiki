---
type: concept
title: Joint Representation Alignment
slug: joint-representation-alignment
date: 2026-04-20
updated: 2026-04-20
aliases: [联合表征对齐]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Joint Representation Alignment** (联合表征对齐) — an alignment strategy that first composes features from multiple modalities into a fused representation and then matches that fused representation against a target modality.

## Key Points

- MixCon3D concatenates image and point-cloud embeddings and projects them with `` `g^(I,P)` `` to form an object-level fused representation.
- The paper optimizes a dedicated contrastive term `` `L^(I,P)<->T` `` so the fused representation aligns directly with text.
- The fused loss is used in addition to, not instead of, the original point-image, point-text, and image-text losses.
- Ablation results show the joint loss alone improves Objaverse-LVIS Top1 from `` `49.8%` `` to `` `51.0%` `` and ScanObjectNN Top1 from `` `55.6%` `` to `` `57.9%` ``.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[gao-2024-sculpting-2311-01734]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[gao-2024-sculpting-2311-01734]].
