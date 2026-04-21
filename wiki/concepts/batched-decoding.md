---
type: concept
title: Batched Decoding
slug: batched-decoding
date: 2026-04-20
updated: 2026-04-20
aliases: []
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Batched Decoding** (批量解码) — decoding multiple sequences together in one forward pass batch so their token generation shares model weight loading.

## Key Points

- SoT uses batched decoding for local open-source models during the point-expanding stage.
- The point-expanding prompts are left-padded so they can be decoded as a single batch.
- Because decoding is bottlenecked by loading weights rather than activations, larger batch sizes produce near-constant per-token latency.
- The paper argues batched decoding enables approximate `B`-fold parallelism when an answer is split into `B` shorter segments.
- Actual gains are lower than the ideal bound because of prefilling overhead and imbalance across point lengths.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[ning-2024-skeletonofthought-2307-15337]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[ning-2024-skeletonofthought-2307-15337]].
