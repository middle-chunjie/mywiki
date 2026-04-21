---
type: concept
title: Query-Aware Video Representation
slug: query-aware-video-representation
date: 2026-04-20
updated: 2026-04-20
aliases: [query aware video embedding, 查询感知视频表征]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Query-Aware Video Representation** (查询感知视频表征) — a video embedding that depends on the current text query by emphasizing frames judged more relevant to that query.

## Key Points

- The paper computes query-aware video features by weighting each frame embedding according to its similarity with the text embedding.
- Temperature-scaled softmax over frame-query scores lets the model interpolate between near-max pooling and near-uniform averaging.
- This representation serves as the final video-side input to the symmetric retrieval loss.
- The method uses the query-aware aggregation to model temporal relevance without adding extra trainable temporal-fusion parameters.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[jiang-2022-crossmodal-2211-09623]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[jiang-2022-crossmodal-2211-09623]].
