---
type: concept
title: Document Reranking
slug: document-reranking
date: 2026-04-20
updated: 2026-04-20
aliases: [reranking, document reranking, 文档重排序]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Document Reranking** (文档重排序) — the task of reordering an initial candidate list produced by a retriever so that more relevant documents move higher in the final ranking.

## Key Points

- The paper studies reranking rather than full retrieval: PRP reorders the top `100` passages returned by BM25 for each query.
- Evaluation is done on TREC-DL2019 and TREC-DL2020 using `NDCG@1`, `NDCG@5`, and `NDCG@10`.
- The proposed reranker is zero-shot and does not fine-tune the LLM on ranking labels.
- PRP reranking with FLAN-UL2 is competitive with or better than strong supervised rerankers such as monoT5 and RankT5 on multiple metrics.
- The paper shows that reranking quality depends not just on model size but also on the prompting formulation and aggregation strategy.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[qin-2024-large-2306-17563]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[qin-2024-large-2306-17563]].
