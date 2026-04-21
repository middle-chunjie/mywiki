---
type: concept
title: Centroid Pruning
slug: centroid-pruning
date: 2026-04-20
updated: 2026-04-20
aliases: [query-aware centroid pruning]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Centroid Pruning** — a query-dependent filtering strategy that discards centroid IDs whose maximum similarity to the query is below a threshold before approximate passage scoring.

## Key Points

- PLAID keeps centroid `i` only when `\max_j S_{c,q_{i,j}} \ge t_{cs}`, using the threshold `t_cs` to sparsify candidate passage representations.
- This pruning happens before centroid interaction, so the system avoids even approximate scoring work for obviously weak centroids.
- The paper uses `t_cs = 0.5`, `0.45`, and `0.4` for final depths `k = 10`, `100`, and `1000` respectively.
- Centroid pruning is especially important for reducing the cost of later stages without materially hurting recall.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[santhanam-2022-plaid-2205-09707]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[santhanam-2022-plaid-2205-09707]].
