---
type: concept
title: Class Hierarchy Distance
slug: class-hierarchy-distance
date: 2026-04-20
updated: 2026-04-20
aliases: [类层级距离, hierarchy distance]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Class Hierarchy Distance** (类层级距离) — the graph distance between two classes in an inheritance hierarchy, used as a proxy for semantic relatedness.

## Key Points

- BLANCA computes shortest-path distances over superclass graphs built from canonicalized Python classes.
- Distances greater than `10` are discarded, yielding a large regression-style training set.
- The task tests whether documentation embeddings reflect structural relatedness between classes.
- Fine-tuned BERTOverflow improves Pearson correlation from `0.17` to `0.34` on this objective.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[abdelaziz-2022-can]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[abdelaziz-2022-can]].
