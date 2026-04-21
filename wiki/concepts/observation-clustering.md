---
type: concept
title: Observation Clustering
slug: observation-clustering
date: 2026-04-20
updated: 2026-04-20
aliases: [step-level observation clustering, state clustering, 观测聚类]
tags: [agents, clustering]
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Observation Clustering** (观测聚类) — grouping semantically similar agent observations into shared clusters so that retrieval and learning signals can be reused across structurally equivalent states.

## Key Points

- SLEA-RL maintains a cluster index `\mathcal{C} = \{c_1, \ldots, c_M\}` where each cluster stores a prototype observation and attached positive/negative experience pools.
- A new observation joins an existing cluster when similarity to a prototype exceeds `\delta = 0.85`; otherwise the system creates a new cluster.
- The same clustering structure supports both cluster-indexed retrieval and within-cluster normalization for step-level advantage estimation.
- The paper motivates clustering as a way to generalize experience across trajectories that encounter equivalent states under different episodes.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[wang-2026-slearl-2603-18079]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[wang-2026-slearl-2603-18079]].
