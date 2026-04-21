---
type: concept
title: Soft Clustering
slug: soft-clustering
date: 2026-04-20
updated: 2026-04-20
aliases: [probabilistic clustering, 软聚类]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Soft Clustering** (软聚类) — a clustering setup in which one item may belong to multiple clusters rather than being assigned to exactly one cluster.

## Key Points

- [[unknown-nd-sirerag]] follows RAPTOR in using Gaussian mixture models to realize soft clustering on the similarity side.
- The relatedness side also approximates soft membership because one proposition can be associated with multiple entity-linked aggregates.
- The paper treats overlapping cluster membership as useful for multihop reasoning, where one evidence unit may support several latent reasoning chains.
- Soft clustering is part of the mechanism that lets higher-level summaries integrate partially overlapping evidence instead of enforcing brittle hard partitions.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[unknown-nd-sirerag]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[unknown-nd-sirerag]].
