---
type: concept
title: Graph-Structured RAG
slug: graph-structured-rag
date: 2026-04-20
updated: 2026-04-20
aliases: [graph structured rag, graph-based rag, 图结构检索增强生成]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Graph-Structured RAG** (图结构检索增强生成) — a retrieval-augmented generation design that organizes knowledge as a graph so retrieval can exploit explicit inter-passage or inter-entity relations instead of relying only on flat similarity search.

## Key Points

- HopRAG instantiates graph-structured RAG with passages as vertices and logical passage-to-passage hops as directed edges.
- The paper argues this is lighter-weight than knowledge-graph RAG because it avoids predefined schemas, triplet textualization, and extra graph-construction error sources.
- Edge semantics are carried by pseudo-queries plus sparse and dense representations, making the graph directly usable for both retrieval and reasoning.
- The resulting graph is intentionally traversal-friendly, with only `O(n log n)` retained edges and an average of `5.87` directed edges per vertex across the evaluated datasets.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[liu-2025-hoprag-2502-12442]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[liu-2025-hoprag-2502-12442]].
