---
type: concept
title: Multi-Vector Retrieval
slug: multi-vector-retrieval
date: 2026-04-20
updated: 2026-04-20
aliases: [multiple-vector retrieval, 多向量检索]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Multi-Vector Retrieval** (多向量检索) — a dense retrieval paradigm that represents a query and a document with multiple token-level vectors and scores them through fine-grained token interactions rather than a single global embedding.

## Key Points

- The paper treats ColBERT-style retrieval as the canonical multi-vector setup, where each query token interacts with many document tokens.
- Multi-vector retrieval improves expressivity over dual encoders because it can preserve token-level evidence alignment.
- The main systems bottleneck is that non-linear token aggregation prevents direct document-level MIPS over exact scores.
- XTR argues that the first-stage token retriever, not only the final reranker, should be optimized as a core part of multi-vector retrieval.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[lee-nd-rethinking]]
- [[wu-2024-generative-2404-00684]]
- [[wu-2024-stark-2404-13207]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[lee-nd-rethinking]].
