---
type: concept
title: Attribute Network Embedding
slug: attribute-network-embedding
date: 2026-04-20
updated: 2026-04-20
aliases: [属性网络嵌入]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Attribute Network Embedding** (属性网络嵌入) — a graph embedding approach that learns node representations from both network structure and node attributes.

## Key Points

- The paper augments citation-graph context with a `742`-dimensional patent attribute vector.
- Each patent input embedding is defined as `e_k = E^T f_k`, so metadata directly enters the graph embedding model.
- Context vectors are averaged from surrounding patents in sampled paths, then optimized with negative sampling.
- This branch injects claims, citations, CPC groups, and trend signals into the final patent representation.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[liu-2018-patent]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[liu-2018-patent]].
