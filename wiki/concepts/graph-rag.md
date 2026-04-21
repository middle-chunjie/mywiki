---
type: concept
title: GraphRAG
slug: graph-rag
date: 2026-04-20
updated: 2026-04-20
aliases: [GraphRAG, graph retrieval-augmented generation]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**GraphRAG** — a retrieval-augmented generation paradigm that retrieves evidence by traversing a knowledge graph built from extracted entities and relations rather than by flat dense retrieval alone.

## Key Points

- In this benchmark, GraphRAG extracts query entities, links them to graph nodes, and expands a salient subgraph with Personalized PageRank.
- The implementation uses up to `20` seed entities, PPR damping `` `\alpha = 0.85` ``, up to `100` retained nodes, and at most `500` serialized triplets.
- GraphRAG performs best when corpora expose explicit entity-centric structure, such as MuSiQue factual questions where it reaches `90.2%` accuracy with DeepSeek-V3.
- The same mechanism degrades on sparse or narrative corpora, showing that graph retrieval is highly sensitive to corpus topology quality.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[wang-2026-ragrouterbench-2602-00296]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[wang-2026-ragrouterbench-2602-00296]].
