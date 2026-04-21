---
type: concept
title: Community Detection
slug: community-detection
date: 2026-04-20
updated: 2026-04-20
aliases: [community discovery, graph partitioning]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Community Detection** (社区发现) — the process of partitioning a graph into densely connected groups whose internal connectivity is stronger than their connectivity to the rest of the graph.

## Key Points

- GraphRAG applies hierarchical Leiden community detection to partition the graph index recursively.
- Each hierarchy level provides a mutually exclusive, collectively exhaustive cover of graph nodes.
- The resulting `C0-C3` summary levels trade off scope, detail, and token cost at query time.
- Community detection is the mechanism that lets GraphRAG convert a large graph into parallelizable summary units.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[edge-2024-local-2404-16130]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[edge-2024-local-2404-16130]].
