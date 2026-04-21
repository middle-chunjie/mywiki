---
type: concept
title: Knowledge Graph
slug: knowledge-graph
date: 2026-04-20
updated: 2026-04-20
aliases: [KG, knowledge graph, knowledge-graph]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Knowledge Graph** (知识图谱) — a structured graph representation in which nodes denote entities and edges denote relations, often augmented with textual descriptions or factual attributes.

## Key Points

- GraphRAG builds a knowledge graph from LLM-extracted entities, relationships, and claims over text chunks.
- Node and edge descriptions are aggregated across duplicate extractions, and repeated relationships increase edge weights.
- The paper uses exact string matching for entity matching before downstream graph summarization.
- This graph becomes the indexing structure that supports hierarchical community summaries and global question answering.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[edge-2024-local-2404-16130]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[edge-2024-local-2404-16130]].
