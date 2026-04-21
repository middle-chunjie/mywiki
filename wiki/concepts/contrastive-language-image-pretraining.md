---
type: concept
title: Contrastive Language-Image Pretraining
slug: contrastive-language-image-pretraining
date: 2026-04-20
updated: 2026-04-20
aliases: [CLIP, contrastive language-image pretraining, 对比语言-图像预训练]
tags: [multimodal, contrastive-learning, vision-language, representation-learning]
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Contrastive Language-Image Pretraining** (对比语言-图像预训练) — a self-supervised pretraining objective that aligns image and text representations by maximizing cosine similarity between paired image-caption embeddings and minimizing it for unpaired combinations across a large web-crawled dataset.

## Key Points

- CLIP trains a dual-encoder (visual + text) on ~400 million image-text pairs from the web, using contrastive loss over a batch of pairings.
- The resulting image encoder produces semantically rich embeddings transferable to a wide range of visual tasks without task-specific fine-tuning.
- Img2Loc uses the CLIP image encoder to embed all gallery images; query similarity is measured by inner product of CLIP embeddings, exploiting the learned semantic space for location-relevant feature retrieval.
- The choice of inner product (rather than L2 distance) as similarity reflects the unit-normalized nature of CLIP embeddings.
- CLIP embeddings capture scene-level semantics (landscapes, urban architecture, vegetation) relevant to geographic disambiguation, even though CLIP was not trained for geolocation.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[zhou-2024-imgloc-2403-19584]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[zhou-2024-imgloc-2403-19584]].
