---
type: concept
title: Variational Graph Reconstruction
slug: variational-graph-reconstruction
date: 2026-04-20
updated: 2026-04-20
aliases: [variational graph generation]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Variational Graph Reconstruction** — learning a latent probabilistic representation of graph nodes and reconstructing observed graph structure from samples of those latent variables.

## Key Points

- VGCL estimates a Gaussian posterior for each user or item node, `` `q_\phi(z_i|A,E^0)=N(\mu_i, diag(\sigma_i^2))` ``, instead of producing only deterministic embeddings.
- Node means are inferred with LightGCN-style propagation over the user-item graph, and variances are produced by a one-layer MLP as `` `\sigma = exp(\mu W + b)` ``.
- Graph reconstruction uses an inner-product decoder, `` `p(A_ij=1|z_i,z_j)=sigmoid(z_i^T z_j)` ``, optimized through an ELBO-style objective.
- The reconstructed latent distribution is reused to sample multiple contrastive views, so generation directly supports self-supervised recommendation learning.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[yang-2023-generative-2307-05100]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[yang-2023-generative-2307-05100]].
