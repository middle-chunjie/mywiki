---
type: concept
title: Neural Reranking
slug: neural-reranking
date: 2026-04-20
updated: 2026-04-20
aliases: [神经重排序]
tags: [retrieval, reranking, bert]
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Neural Reranking** (神经重排序) — a retrieval architecture in which a neural model rescoring a candidate set refines the ranking produced by a first-stage retriever.

## Key Points

- The paper uses BM25 as the first-stage retriever and trains a BERT-based top-100 reranker on synthetic question-document pairs.
- The reranker is intentionally simple: `bert-base-uncased` with random negative sampling and mostly default Hugging Face training settings.
- Despite the simple setup, the synthetic-data-trained reranker improves over BM25 on all five evaluation datasets.
- The work argues that cheap synthetic data is sufficient to make neural reranking viable even when no manual labels exist for the target collection.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[almeida-2024-exploring]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[almeida-2024-exploring]].
