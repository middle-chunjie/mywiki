---
type: concept
title: Incremental Index Update
slug: incremental-index-update
date: 2026-04-20
updated: 2026-04-20
aliases: [incremental update, 增量索引更新]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Incremental Index Update** (增量索引更新) — an index maintenance strategy that incorporates newly arrived documents into an existing retrieval structure without rebuilding the full index from scratch.

## Key Points

- For a new document, LightRAG reruns the same graph-based indexer and then unions the new node and edge sets with the existing graph.
- The paper positions this as a core advantage over graph pipelines that require expensive global restructuring when new data arrives.
- Incremental updating is designed to preserve historical graph structure while extending it with new entities and relations.
- The reported cost analysis highlights large savings relative to GraphRAG when corpora evolve over time.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[unknown-nd-lightrag]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[unknown-nd-lightrag]].
