---
type: concept
title: Knowledge Graph Embedding
slug: knowledge-graph-embedding
date: 2026-04-20
updated: 2026-04-20
aliases: [知识图谱嵌入, relational embedding]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Knowledge Graph Embedding** (知识图谱嵌入) — a representation learning technique that maps entities and relations in a relational graph into vectors so that graph structure can be used in downstream models.

## Key Points

- HPM uses TransE to pretrain item and relation embeddings from explicit product relations before sequential modeling.
- The relation embeddings are later reused in SCEL to inject semantics such as complementarity and substitutability into target context embeddings.
- Both item-level and category-level relation triples are encoded with an `L2` translation objective.
- In this paper, knowledge graph embedding is not the final model itself but a preparatory layer that strengthens sparse-sequence recommendation.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[huang-2023-dual]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[huang-2023-dual]].
