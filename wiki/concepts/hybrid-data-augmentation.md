---
type: concept
title: Hybrid Data Augmentation
slug: hybrid-data-augmentation
date: 2026-04-20
updated: 2026-04-20
aliases: [混合数据增强]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Hybrid Data Augmentation** (混合数据增强) — a training-data construction strategy that combines multiple sampling and filtering routes to build candidate lists with both diversity and hard negative pressure.

## Key Points

- [[ren-2023-rocketqav-2110-07367]] uses RocketQA to retrieve top-`n` passages before forming training lists for joint retriever-reranker optimization.
- The method mixes undenoised instances built from randomly sampled hard negatives with denoised instances filtered by the RocketQA re-ranker.
- Denoised positives are also retained when the re-ranker assigns them high confidence, so augmentation can improve both positive and negative coverage.
- The goal is to make each candidate list better approximate the passage distribution of the full collection for listwise training.
- Removing denoised instances lowers MSMARCO retriever `MRR@10` from `37.4` to `36.3`, indicating that the mixed construction matters.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[ren-2023-rocketqav-2110-07367]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[ren-2023-rocketqav-2110-07367]].
