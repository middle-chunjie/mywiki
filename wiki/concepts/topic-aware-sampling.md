---
type: concept
title: Topic Aware Sampling
slug: topic-aware-sampling
date: 2026-04-20
updated: 2026-04-20
aliases: [TAS, 主题感知采样]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Topic Aware Sampling** (主题感知采样) — a batch-construction strategy that groups training queries by semantic topic and samples each mini-batch from one or a few topical clusters to increase the usefulness of in-batch interactions.

## Key Points

- The paper encodes roughly `400K` MS MARCO training queries once with a baseline dense retriever and clusters them with `k`-means into `2000` groups.
- With the default setup, each batch of size `32` is drawn from `n = 1` cluster rather than uniformly from the whole query pool.
- Concentrating semantically related queries in one batch makes in-batch negatives more informative than fully random batch composition.
- The one-time clustering step costs under `10` minutes, so the method improves training signal without adding per-step retrieval overhead.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[hofst-tter-2021-efficiently-2104-06967]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[hofst-tter-2021-efficiently-2104-06967]].
