---
type: concept
title: Image Embedding
slug: image-embedding
date: 2026-04-20
updated: 2026-04-20
aliases: [visual embedding, 图像嵌入]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Image Embedding** (图像嵌入) — a vector representation of an image constructed so that geometric similarity in the embedding space reflects semantic or instance-level visual similarity.

## Key Points

- [[wu-2023-forb-2309-16249]] studies whether image embeddings trained on common retrieval domains remain effective on diverse flat 2D objects.
- The paper evaluates embeddings from low-level handcrafted features, mid-level learned local features, and high-level global semantic models under the same retrieval protocol.
- Embedding quality is assessed not only by ranking accuracy (`mAP`) but also by margin against false positives via `t-mAP`.
- Results show that mid-level embeddings such as FIRe can generalize strongly even when trained on 3D landmark data, while high-level semantic embeddings are better on feature-scarce categories such as logos.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[wu-2023-forb-2309-16249]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[wu-2023-forb-2309-16249]].
