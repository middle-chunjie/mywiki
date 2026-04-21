---
type: concept
title: Neural Information Retrieval
slug: neural-ir
date: 2026-04-20
updated: 2026-04-20
aliases: [neural IR, neural retrieval, neural ranking, deep IR]
tags: [information-retrieval, neural-network, nlp]
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Neural Information Retrieval** (神经信息检索) — the paradigm of using neural networks to model query-document relevance, replacing or augmenting traditional term-based methods such as BM25 by learning dense semantic representations or interaction features.

## Key Points

- Neural IR encompasses both representation-based approaches (bi-encoders that embed queries and documents into vector spaces) and interaction-based approaches (cross-encoders that directly attend over the query–document pair).
- The paradigm shift from BM25 to neural models enables capturing semantic nuances and long-range contextual signals that keyword matching misses, at the cost of significantly higher compute.
- Foundation models (BERT, T5, LLaMA) have become the dominant backbone for neural retrievers and rerankers; model scale consistently improves performance, especially in zero-shot transfer.
- Key challenges include latency (neural models are orders of magnitude slower than inverted index lookups), data scarcity in specialized domains, and limited interpretability.
- Retrieval is typically structured as a two-stage pipeline: fast neural or BM25 first-pass recall followed by a computationally intensive neural reranker.

## My Position

<!-- User's stance on this concept. Fed by personal writing; tag "(personal stance)" on such bullets. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[zhu-2024-large-2308-07107]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[zhu-2024-large-2308-07107]].
