---
type: concept
title: Cross-Batch Negative Sampling
slug: cross-batch-negative-sampling
date: 2026-04-20
updated: 2026-04-20
aliases: [cross-batch negatives, 跨批负采样]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Cross-Batch Negative Sampling** (跨批负采样) — a retrieval training strategy that enlarges each query's negative set by reusing candidate representations from other mini-batches or devices as negatives.

## Key Points

- RocketQA extends standard in-batch negatives by sharing passage embeddings across GPUs with an all-gather operation.
- With `A` GPUs and `B` questions per GPU, each question can see `A × B - 1` negatives instead of only `B - 1`.
- The method targets the discrepancy between training on small batches and inference over millions of passages.
- RocketQA applies cross-batch negatives throughout its dual-encoder training pipeline, not only in the first stage.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[qu-2021-rocketqa-2010-08191]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[qu-2021-rocketqa-2010-08191]].
