---
type: concept
title: Hybrid Retrieval
slug: hybrid-retrieval
date: 2026-04-20
updated: 2026-04-20
aliases: [dense-lexical retrieval, mixed retrieval, 混合检索]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Hybrid Retrieval** (混合检索) — a retrieval strategy that combines lexical and dense retrieval signals to construct or rank the final evidence set.

## Key Points

- In the appendix QA pipeline, the paper combines top-`3` BM25 chunks with top-`15` dense-retrieval chunks to balance exact mention matching and semantic recall.
- BM25 evidence is retained preferentially when the query mentions people or events, reflecting the importance of exact lexical grounding in biography QA.
- Overlapping results between lexical and dense retrieval are filtered to avoid redundant context.
- Retrieved chunks are reordered before final generation, connecting hybrid retrieval to long-context prompt management rather than ranking alone.
- LumberChunker is evaluated not only as a chunker for retrieval metrics but also as a component inside this broader hybrid RAG stack.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[duarte-2024-lumberchunker-2406-17526]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[duarte-2024-lumberchunker-2406-17526]].
