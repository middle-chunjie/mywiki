---
type: concept
title: Graph-Based Indexing
slug: graph-based-indexing
date: 2026-04-20
updated: 2026-04-20
aliases: [graph indexing, 图结构索引]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Graph-Based Indexing** (图结构索引) — an indexing strategy that organizes a corpus as entities and relations in a graph, often with attached textual summaries, so retrieval can exploit structural dependencies rather than isolated chunks alone.

## Key Points

- LightRAG builds its index as a knowledge graph over chunk-level entity and relation extractions instead of a flat chunk store.
- The index is produced by entity-relation recognition, LLM profiling into key-value text summaries, and deduplication across chunks.
- This structure is intended to preserve cross-chunk and multi-hop dependencies needed for complex questions.
- The paper argues graph-based indexing improves both retrieval comprehensiveness and efficiency compared with brute-force chunk traversal.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[unknown-nd-lightrag]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[unknown-nd-lightrag]].
