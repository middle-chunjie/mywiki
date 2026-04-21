---
type: concept
title: Clustering
slug: clustering
date: 2026-04-20
updated: 2026-04-20
aliases: [cluster analysis, 聚类]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Clustering** (聚类) — grouping similar representations into shared regions or states so that downstream algorithms can operate on aggregated structure rather than isolated instances.

## Key Points

- RETOMATON clusters datastore keys into automaton states and uses those shared states to support search-free continuation.
- The paper uses `k`-means as the main clustering method with `k_clus = 1M` for WIKIText-103 and `200K` for Law-MT.
- Average cluster size is kept around `100`, balancing state granularity against traversal stability.
- Ablations show clustering matters most at high FoSS, where longer no-search segments depend on broader state coverage.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[alon-2022-neurosymbolic-2201-12431]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[alon-2022-neurosymbolic-2201-12431]].
