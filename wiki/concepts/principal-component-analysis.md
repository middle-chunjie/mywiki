---
type: concept
title: Principal Component Analysis
slug: principal-component-analysis
date: 2026-04-20
updated: 2026-04-20
aliases: [PCA, 主成分分析]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Principal Component Analysis** (主成分分析) — a linear projection method that maps vectors into lower-dimensional orthogonal components ordered by explained variance.

## Key Points

- The paper uses PCA to compress datastore vectors before nearest-neighbor retrieval.
- FAISS's PCA implementation is used, including its default random rotation that can rebalance component variance for indexing.
- On WikiText-103, reducing vectors to `512` dimensions improves speed from `277` to `991` tokens/s while also improving perplexity from `16.65` to `16.40`.
- The results suggest PCA can produce a retrieval space that is better aligned with `L2` distance than the original representation.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[he-2021-efficient-2109-04212]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[he-2021-efficient-2109-04212]].
