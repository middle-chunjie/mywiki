---
type: concept
title: Personalized PageRank
slug: personalized-pagerank
date: 2026-04-20
updated: 2026-04-20
aliases: [PPR, personalized PageRank, 个性化 PageRank]
tags: []
source_count: 1
confidence: low
domain_volatility: low
last_reviewed: 2026-04-20
---

## Definition

**Personalized PageRank** (个性化 PageRank) — a graph ranking algorithm that propagates probability mass from a user-defined seed distribution so scores concentrate around nodes connected to the chosen sources.

## Key Points

- HippoRAG uses query-linked KG nodes as the personalized seed set, giving non-query nodes zero initial probability.
- The graph contains both extracted relation edges `E` and synonymy edges `E'`, so PPR can move through factual and approximate-semantic associations.
- The output node distribution `n'` is projected back to passages through `p = n' P`, turning graph relevance into passage-level retrieval scores.
- The tuned damping factor is `0.5`, balancing restarts from the query nodes against deeper graph exploration.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[guti-rrez-2024-hipporag-2405-14831]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[guti-rrez-2024-hipporag-2405-14831]].
