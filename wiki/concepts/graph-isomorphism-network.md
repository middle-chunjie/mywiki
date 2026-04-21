---
type: concept
title: Graph Isomorphism Network
slug: graph-isomorphism-network
date: 2026-04-20
updated: 2026-04-20
aliases: [GIN, 图同构网络]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Graph Isomorphism Network** (图同构网络) — a graph neural network architecture that uses sum aggregation and MLP-based updates to approximate the expressive power of the Weisfeiler-Lehman test for graph discrimination.

## Key Points

- The paper uses a `5`-layer GIN with hidden dimension `32` in its preliminary semantic-similarity investigation over sampled subgraphs.
- In the unsupervised benchmark setting, MSSGCL adopts GIN as the encoder backbone together with sum pooling.
- GIN is used because the work focuses on graph-level discrimination where expressive neighborhood aggregation matters.
- The paper positions GIN against other GNN families such as GCN and GAT in its related-work discussion.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[liu-2023-multiscale]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[liu-2023-multiscale]].
