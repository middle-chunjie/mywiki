---
type: concept
title: Vision Transformer
slug: vision-transformer
date: 2026-04-20
updated: 2026-04-20
aliases: [ViT, Vision Transformer, 视觉Transformer]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Vision Transformer** (视觉Transformer) — a Transformer-based image encoder (Dosovitskiy et al., 2021) that treats an image as a sequence of fixed-size non-overlapping patches linearly projected into token embeddings, enabling standard self-attention to model global spatial dependencies without convolutions.

## Key Points

- Each image is split into `P × P` patches (typically 16×16 pixels), flattened, and linearly projected to a `d`-dimensional token sequence, prepended by a learnable `[CLS]` token whose final representation is used for classification.
- Pre-training on large-scale image datasets (e.g., ImageNet-21K, JFT) is essential; ViT-Base trained only on ImageNet shows weaker inductive bias than ResNets of comparable scale.
- In UniS-MMC, pretrained ViT-Base is used as the image encoder for both UPMC-Food-101 and N24News; it achieves 73.1% image-only accuracy on N24News.
- ViT provides a natural counterpart to BERT/RoBERTa in multimodal frameworks: both use Transformer architectures and produce token-level representations suitable for cross-modal fusion or contrastive alignment.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[zou-2023-unismmc-2305-09299]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[zou-2023-unismmc-2305-09299]].
