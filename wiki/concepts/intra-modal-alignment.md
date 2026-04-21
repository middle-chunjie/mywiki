---
type: concept
title: Intra-Modal Alignment
slug: intra-modal-alignment
date: 2026-04-20
updated: 2026-04-20
aliases: [模态内对齐, intra-modality alignment, intra-modality consistency]
tags: [contrastive-learning, multimodal, representation-learning]
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Intra-Modal Alignment** (模态内对齐) — an additional contrastive objective that enforces semantically similar items within the same modality to remain proximate in the joint embedding space, complementing the inter-modal alignment that only pairs items across modalities.

## Key Points

- Standard cross-modal contrastive losses only enforce inter-modality alignment (video ↔ text) and implicitly assume transitivity will preserve within-modality similarity; CrossCLR shows empirically this assumption fails.
- CrossCLR implements intra-modal alignment by adding intra-modality negative terms to the contrastive loss (Eq. 2–3): for query `x_i` the intra-modal negative set is `N^R_i = {x_j | j ≠ i}`, weighted by hyperparameter `λ`.
- The intra-modal component `I_M` is the single largest contributor in the CrossCLR ablation study, yielding +0.8 pp R@1 Text→Video and +1.0 pp R@5 over the baseline on Youcook2.
- Intra-modal alignment is only meaningful when input embeddings already carry semantic structure (e.g., from ImageNet-pretrained features), distinguishing this paradigm from scratch-trained representation learning.
- t-SNE visualizations show CrossCLR embeddings have higher intra-modality consistency compared to NT-Xent (SimCLR) baselines.

## My Position

<!-- User's stance on this concept. Fed by personal writing; tag "(personal stance)" on such bullets. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[zolfaghari-2021-crossclr-2109-14910]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[zolfaghari-2021-crossclr-2109-14910]].
