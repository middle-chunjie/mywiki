---
type: concept
title: Multi-Stage Retrieval
slug: multi-stage-retrieval
date: 2026-04-20
updated: 2026-04-20
aliases: [two-stage retrieval, 多阶段检索]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Multi-Stage Retrieval** (多阶段检索) — a retrieval architecture that first retrieves a manageable candidate set efficiently and then applies one or more more expensive ranking stages to improve final relevance.

## Key Points

- The paper instantiates multi-stage retrieval as a dense retriever plus a pointwise reranker rather than a sparse first stage plus cross-encoder reranker.
- `RepLLaMA` produces the top `1000` candidates and `RankLLaMA` reranks the top `200` passages or top `100` documents.
- The reranker is trained on negatives sampled from the paired retriever, aligning downstream scoring with the actual upstream candidate distribution.
- The paper argues that prompt-based LLM reranking does not optimize the whole pipeline, whereas fine-tuning retriever and reranker does.
- The resulting pipeline outperforms prior end-to-end systems on both passage and document retrieval benchmarks.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[ma-2023-finetuning-2310-08319]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[ma-2023-finetuning-2310-08319]].
