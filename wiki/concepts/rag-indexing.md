---
type: concept
title: RAG Indexing
slug: rag-indexing
date: 2026-04-20
updated: 2026-04-20
aliases: [retrieval indexing for rag, 检索增强生成索引构建]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**RAG Indexing** (检索增强生成索引构建) — the stage of a retrieval-augmented generation pipeline that organizes raw evidence into a retrieval structure before query-time search.

## Key Points

- [[unknown-nd-sirerag]] argues indexing is not just storage management but a knowledge-integration step that can materially change downstream multihop QA accuracy.
- The paper contrasts similarity-oriented indexing such as RAPTOR with relatedness-oriented indexing such as GraphRAG and HippoRAG.
- SiReRAG builds separate similarity and relatedness trees, then flattens both into one retrieval pool.
- The reported gains suggest that indexing structure can improve both dense retrieval and reranking-based pipelines.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[unknown-nd-sirerag]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[unknown-nd-sirerag]].
