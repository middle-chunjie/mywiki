---
type: concept
title: Sparse Retrieval
slug: sparse-retrieval
date: 2026-04-20
updated: 2026-04-20
aliases: [稀疏检索, learned sparse retrieval]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Sparse Retrieval** (稀疏检索) — a retrieval paradigm that represents queries and documents with sparse weighted features, often learned, to preserve interpretable term-level matching while improving semantic coverage.

## Key Points

- mtRAG uses Elser as the paper's sparse retriever in both benchmark construction and retrieval experiments.
- Sparse retrieval is the best-performing retrieval family in the paper, outperforming BM25 and BGE-base 1.5 under both last-turn and rewritten-query settings.
- With query rewriting, Elser reaches `R@5 = 0.52` and `nDCG@5 = 0.48` on the overall benchmark.
- Despite its advantage, sparse retrieval still degrades sharply on later turns and non-standalone questions, showing that multi-turn RAG remains difficult.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[katsis-2025-mtrag-2501-03468]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[katsis-2025-mtrag-2501-03468]].
