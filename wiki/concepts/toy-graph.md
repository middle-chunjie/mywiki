---
type: concept
title: Toy Graph
slug: toy-graph
date: 2026-04-20
updated: 2026-04-20
aliases: [toy graph, toy graphs]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Toy Graph** - a compact retrieved subgraph centered on a sampled master node and stored as a reusable key-value memory unit for downstream graph inference.

## Key Points

- RAGraph chunks each resource graph into small `k`-hop ego graphs and treats them as toy graphs.
- Each toy graph stores keys for time, neighborhood environment, structure encoding, and semantic embedding of the master node.
- Values include hidden embeddings and task-specific output vectors for nodes inside the toy graph.
- Retrieved toy graphs are linked to the query graph and used for both intra-graph and cross-graph message passing.
- The paper argues toy graphs make graph-native retrieval practical without retrieving whole training graphs.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[jiang-2024-ragraph-2410-23855]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[jiang-2024-ragraph-2410-23855]].
