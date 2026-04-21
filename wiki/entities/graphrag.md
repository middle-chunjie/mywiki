---
type: entity
title: GraphRAG
slug: graphrag
date: 2026-04-20
entity_type: tool
aliases: [GraphRAG, Graph RAG]
tags: []
---

## Description

GraphRAG is the graph-based retrieval-augmented generation system introduced in [[edge-2024-local-2404-16130]]. It answers global corpus questions by indexing text as a knowledge graph and summarizing graph communities.

## Key Contributions

- Builds a graph index from LLM-extracted entities, relationships, and claims.
- Uses hierarchical Leiden communities plus precomputed community summaries for scalable global reasoning.
- Outperforms a vector RAG baseline on comprehensiveness and diversity for corpus-scale sensemaking questions.

## Related Concepts

- [[retrieval-augmented-generation]]
- [[knowledge-graph]]
- [[community-detection]]
- [[map-reduce-summarization]]

## Sources

- [[edge-2024-local-2404-16130]]
