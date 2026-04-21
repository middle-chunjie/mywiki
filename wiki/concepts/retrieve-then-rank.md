---
type: concept
title: Retrieve-Then-Rank
slug: retrieve-then-rank
date: 2026-04-20
updated: 2026-04-20
aliases: [retrieval-then-reranking, two-stage retrieval, 检索再排序]
tags: [information-retrieval, pipeline, dense-retrieval, reranking]
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Retrieve-Then-Rank** (检索再排序) — a two-stage information retrieval pipeline where a fast first-stage retriever (typically a dual-encoder) returns a candidate set, which a slower but more accurate second-stage ranker (typically a cross-encoder) then re-scores to produce the final ranked list.

## Key Points

- The first stage prioritizes recall and speed: a dual-encoder computes dense embeddings offline and uses ANN search at query time, returning top-k candidates (e.g., k=100) in milliseconds.
- The second stage prioritizes precision: a cross-encoder performs full query-document interaction (e.g., via self-attention over the concatenation), producing more accurate relevance scores at the cost of being unable to pre-compute document representations.
- The pipeline decouples the efficiency-accuracy tradeoff: scalability is handled by the retriever while accuracy is handled by the ranker.
- AR2 demonstrates that the two stages can be jointly optimized via an adversarial minimax objective, with the retriever generating progressively harder negatives for ranker training and the ranker providing training signal back to the retriever.
- Classical IR systems (BM25 + BERT cross-encoder) use this pipeline with independently trained components; joint training as in AR2 enables mutual improvement.

## My Position

<!-- User's stance on this concept. Fed by personal writing; tag "(personal stance)" on such bullets. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[zhang-2022-adversarial]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[zhang-2022-adversarial]].
