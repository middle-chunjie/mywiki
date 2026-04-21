---
type: concept
title: Product Search
slug: product-search
date: 2026-04-20
updated: 2026-04-20
aliases: [shopping search, 商品搜索]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Product Search** (商品搜索) — the retrieval task of ranking products for a shopping query based on semantic relevance between the query and product representations.

## Key Points

- The paper evaluates product retrieval on ESCI, where product descriptions are the structured documents and bullet points provide aligned natural-language text for pretraining.
- ESCI annotations include four relevance levels: Exact, Substitute, Complement, and Irrelevant.
- SANTA is evaluated under both a two-class setting and the original four-class graded-relevance setting using `NDCG@100`.
- Structure-aware pretraining improves product-search performance in both zero-shot and fine-tuned regimes relative to plain T5 baselines.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[li-2023-structureaware-2305-19912]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[li-2023-structureaware-2305-19912]].
