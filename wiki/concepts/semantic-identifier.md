---
type: concept
title: Semantic Identifier
slug: semantic-identifier
date: 2026-04-20
updated: 2026-04-20
aliases: [Semantic ID, 语义标识符]
tags: [retrieval, clustering]
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Semantic Identifier** (语义标识符) — a hierarchical document identifier whose token sequence reflects recursive clustering structure in document embedding space rather than an arbitrary external id.

## Key Points

- The paper builds Semantic IDs by recursively clustering document embeddings with hierarchical `k`-means until leaf clusters contain at most `c = 100` documents.
- For MS MARCO, the clustering uses `k = 10` at each level and SentenceT5-Base embeddings, with sampling used when clusters exceed `1M` documents during centroid estimation.
- Plain Semantic IDs degrade at large corpus scale and underperform Naive IDs on MSMarcoFULL, suggesting that longer structured identifiers are harder to decode robustly.
- The paper also studies 2D Semantic IDs with PAWA, but finds that the added decoder complexity still fails to beat naively scaled Naive-ID models.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[pradeep-2023-how]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[pradeep-2023-how]].
