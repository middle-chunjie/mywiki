---
type: concept
title: Nearest-Neighbor Retrieval
slug: nearest-neighbor-retrieval
date: 2026-04-20
updated: 2026-04-20
aliases: [nearest neighbor lookup, 最近邻检索]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Nearest-Neighbor Retrieval** (最近邻检索) — a retrieval strategy that returns the most similar items in an embedding space according to a distance or similarity metric such as cosine similarity.

## Key Points

- ReFIR retrieves the top-`k` high-quality reference images from a fixed database using cosine similarity between the low-quality query image and precomputed feature vectors.
- The paper emphasizes that the feature extractor can be any pretrained image encoder, explicitly naming `VGG`, `ResNet`, and `CLIP`.
- Because the retrieval database is fixed, compact features can be pre-extracted and stored offline to keep lookup efficient at inference time.
- The authors intentionally use semantic retrieval as a simple non-parametric baseline and leave texture-specialized retrievers to future work.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[guo-2024-refir-2410-05601]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[guo-2024-refir-2410-05601]].
