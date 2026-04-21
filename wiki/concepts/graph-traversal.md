---
type: concept
title: Graph Traversal
slug: graph-traversal
date: 2026-04-20
updated: 2026-04-20
aliases: [graph traversal, 图遍历]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Graph Traversal** (图遍历) — the process of exploring graph nodes and edges according to a search policy in order to discover relevant structure or accumulate task-specific evidence.

## Key Points

- HopRAG uses breadth-first local graph traversal rather than a one-shot top-`k` retrieval list.
- For each visited vertex, an LLM evaluates outgoing pseudo-queries and chooses one neighbor judged most helpful for the user query.
- The traversal keeps a visit counter `C_count`, treating repeated visits as a signal that a vertex is logically important.
- After `n_hop` rounds, traversal outputs are pruned by a Helpfulness score that combines similarity to the query with traversal-derived importance.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[liu-2025-hoprag-2502-12442]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[liu-2025-hoprag-2502-12442]].
