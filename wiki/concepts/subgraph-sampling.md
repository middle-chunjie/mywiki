---
type: concept
title: Subgraph Sampling
slug: subgraph-sampling
date: 2026-04-20
updated: 2026-04-20
aliases: [子图采样]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Subgraph Sampling** (子图采样) — an augmentation procedure that selects a subset of nodes and edges from a graph to form a smaller view while preserving part of the original structure.

## Key Points

- MSSGCL uses random-walk-based subgraph sampling as its primary graph augmentation mechanism.
- The method controls the number of sampled nodes to create two scales of views: global subgraphs and local subgraphs.
- The paper's empirical analysis shows that semantic similarity increases as sampled subgraphs become larger.
- Two sampled global views and two sampled local views are generated for each anchor graph to support three contrastive relations.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[liu-2023-multiscale]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[liu-2023-multiscale]].
