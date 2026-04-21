---
type: concept
title: Lexical Retrieval
slug: lexical-retrieval
date: 2026-04-20
updated: 2026-04-20
aliases: [词法检索, term-based retrieval]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Lexical Retrieval** (词法检索) — a retrieval approach that ranks documents primarily through lexical term overlap and weighting rather than dense semantic embedding similarity.

## Key Points

- mtRAG evaluates BM25 as its lexical retrieval baseline for multi-turn RAG.
- The paper shows lexical retrieval benefits from query rewriting, improving from `R@5 = 0.20` to `0.25`.
- Even with rewriting, BM25 remains far below sparse Elser retrieval on the benchmark.
- The results highlight that exact-term matching alone struggles on later-turn and non-standalone conversational questions.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[katsis-2025-mtrag-2501-03468]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[katsis-2025-mtrag-2501-03468]].
