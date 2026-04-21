---
type: concept
title: Graph Attention Network
slug: graph-attention-network
date: 2026-04-20
updated: 2026-04-20
aliases: [GAT, 图注意力网络]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Graph Attention Network** (图注意力网络) — a graph neural network that learns attention weights over graph neighbors so each node can aggregate neighborhood information non-uniformly.

## Key Points

- [[jin-2023-code]] uses a `3`-layer GAT on the repository hierarchy graph to refine file, directory, and repository representations.
- The GAT stage operates after code-user attention, so it injects structural context into already semantic-aware file embeddings.
- Directory and repository nodes are mapped into the same latent space before GAT propagation.
- The paper contrasts GAT with alternative GNN choices such as GCN, GraphSAGE, and GIN, but selects GAT in practice.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[jin-2023-code]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[jin-2023-code]].
