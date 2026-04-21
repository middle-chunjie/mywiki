---
type: concept
title: Graph Convolutional Network
slug: graph-convolutional-network
date: 2026-04-20
updated: 2026-04-20
aliases: [图卷积网络, GCN]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Graph Convolutional Network** (图卷积网络) — a graph neural network that updates node representations by aggregating information from local neighborhoods through layered message passing.

## Key Points

- RHGN is built on a stacked GCN-style framework so entity embeddings absorb higher-order neighbor information across layers.
- The paper restates the generic update as an aggregation over neighbor messages `Agg_{j∈N(i)} φ(e_i, e_j, r_ij)` followed by a transformation `γ`.
- Prior GCN variants for entity alignment either ignore relations or use them in ways that insufficiently separate relation and entity semantics.
- RHGN's relation-gated convolution is presented as a specialized graph convolution for knowledge graphs with heterogeneous relation structure.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[liu-2023-rhgn]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[liu-2023-rhgn]].
