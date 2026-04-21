---
type: concept
title: Nearest-Neighbor Search
slug: nearest-neighbor-search
date: 2026-04-20
updated: 2026-04-20
aliases: [kNN search, nearest neighbor retrieval, 最近邻搜索]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Nearest-Neighbor Search** (最近邻搜索) — retrieval of the most similar vectors to a query representation under a chosen distance metric, typically used here to access external LM memory.

## Key Points

- The paper identifies nearest-neighbor search as the dominant inference cost in retrieval-based language models because it is repeated at every token.
- RETOMATON does not eliminate search entirely; it triggers new search only when the active state set falls below threshold `\tau`.
- The reported implementation uses FAISS and retrieves `1024` neighbors during each full search.
- Fraction of Saved Searches (FoSS) is the main control axis and evaluation proxy for how often full search is avoided.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[alon-2022-neurosymbolic-2201-12431]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[alon-2022-neurosymbolic-2201-12431]].
