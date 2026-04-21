---
type: concept
title: Graph Neural Network
slug: graph-neural-network
date: 2026-04-20
updated: 2026-04-20
aliases: [GNN, 图神经网络]
tags: [graph-learning, neural-network]
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Graph Neural Network** (图神经网络) — a neural architecture for graph-structured data that propagates information over graph neighborhoods and transforms node or graph representations across layers.

## Key Points

- MA-GCL decomposes a GNN into propagation operators `g` and transformation operators `h` to expose architectural degrees of freedom for augmentation.
- The paper writes `g(Z; F) = FZ` and `h(Z; W) = sigma(ZW)`, making the graph filter `F` explicit.
- Standard architectures such as GCN and SGC appear as different compositions of the same operator primitives.
- MA-GCL keeps parameters shared across views while varying propagation depth and operator order.
- The method treats encoder architecture itself as a controllable source of contrastive view diversity.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[gong-2022-magcl-2212-07035]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[gong-2022-magcl-2212-07035]].
