---
type: concept
title: Prize-Collecting Steiner Tree
slug: prize-collecting-steiner-tree
date: 2026-04-20
updated: 2026-04-20
aliases: [PCST, prize collecting steiner tree]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Prize-Collecting Steiner Tree** — an optimization problem that selects a connected subgraph by maximizing collected node or edge prizes while paying a cost for included edges.

## Key Points

- G-Retriever uses PCST to turn top-`k` retrieved nodes and edges into a connected subgraph for downstream question answering.
- Retrieved nodes and edges receive descending prize values based on cosine-similarity rank, while edge cost `` `C_e` `` controls the allowed subgraph size.
- The objective optimized in the paper is `` `sum prize(nodes) + sum prize(edges) - cost(S)` `` over connected subgraphs.
- To support edge prizes, the method replaces an edge with a virtual node when the edge prize exceeds its cost, allowing the solver to stay within the standard PCST formulation.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[he-2024-gretriever-2402-07630]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[he-2024-gretriever-2402-07630]].
