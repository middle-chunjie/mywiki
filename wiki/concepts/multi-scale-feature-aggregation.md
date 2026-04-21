---
type: concept
title: Multi-Scale Feature Aggregation
slug: multi-scale-feature-aggregation
date: 2026-04-20
updated: 2026-04-20
aliases: [multi-scale pooling, 多尺度特征聚合]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Multi-Scale Feature Aggregation** (多尺度特征聚合) — the construction of a single retrieval representation by extracting features from multiple image scales and combining them into one normalized embedding.

## Key Points

- [[wu-2023-forb-2309-16249]] uses multi-scale inference for both top-only and local-feature-based retrieval baselines.
- Query images use `3` scales `{1/√2, 1, √2}`, while database images use `7` scales.
- Features are `L2`-normalized at each scale, average-pooled across scales, and normalized again after aggregation.
- The paper reports that this pipeline improves retrieval accuracy relative to single-scale representations by reducing scale-sensitivity.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[wu-2023-forb-2309-16249]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[wu-2023-forb-2309-16249]].
