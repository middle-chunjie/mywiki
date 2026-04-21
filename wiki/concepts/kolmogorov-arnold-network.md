---
type: concept
title: Kolmogorov-Arnold Network
slug: kolmogorov-arnold-network
date: 2026-04-20
updated: 2026-04-20
aliases: [KAN, 柯尔莫哥洛夫-阿诺德网络]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Kolmogorov-Arnold Network** (柯尔莫哥洛夫-阿诺德网络) — a neural architecture that replaces scalar edge weights with learnable univariate functions and sums their transformed signals at each node.

## Key Points

- The paper defines a KAN layer as a matrix of edge functions `\phi_{l,j,i}`, with node updates `x_{l+1,j} = \sum_i \phi_{l,j,i}(x_{l,i})`.
- Practical KANs parameterize each edge as a residual base function plus a spline term, `\phi(x) = w_b b(x) + w_s spline(x)`, rather than using fixed node activations.
- The architecture generalizes the original depth-2 Kolmogorov-Arnold construction to arbitrary widths and depths, represented by shapes such as `[2,1,1]` or `[17,1,14]`.
- Grid extension lets a trained KAN increase spline resolution without restarting training, which is a major part of its empirical scaling behavior.
- The paper positions KANs as especially useful for small-scale AI-for-science tasks where [[interpretability]] and compact function decompositions are important.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[liu-2024-kan-2404-19756]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[liu-2024-kan-2404-19756]].
