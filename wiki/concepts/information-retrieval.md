---
type: concept
title: Information Retrieval
slug: information-retrieval
date: 2026-04-20
updated: 2026-04-20
aliases: [IR, 信息检索]
tags: [ir, retrieval]
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Information Retrieval** (信息检索) — the problem of retrieving relevant documents or passages from a collection in response to a user information need expressed as a query.

## Key Points

- This paper frames synthetic question generation as a way to build training data for IR collections that lack manual relevance labels.
- The target use case is document retrieval and reranking over large unlabeled collections rather than question answering alone.
- BM25 and monoT5 are used as retrieval-oriented filters to decide whether a generated question-document pair is useful for IR training.
- The downstream evaluation measures whether synthetic data improves IR metrics such as `nDCG@10` and `MRR@10` over an unsupervised BM25 baseline.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[almeida-2024-exploring]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[almeida-2024-exploring]].
