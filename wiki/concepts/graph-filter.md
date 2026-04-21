---
type: concept
title: Graph Filter
slug: graph-filter
date: 2026-04-20
updated: 2026-04-20
aliases: [图滤波器]
tags: [graph-learning, spectral-method]
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Graph Filter** (图滤波器) — a matrix operator derived from graph structure that propagates and smooths signals over nodes, often interpreted spectrally through its eigenvalues and eigenvectors.

## Key Points

- MA-GCL uses the filter `F = (1 - pi)I + pi D^(-1/2) A D^(-1/2)` with `pi = 0.5`.
- The paper studies the asymmetric strategy via the eigendecomposition `F = U Lambda U^T`.
- Larger eigenvalues of `F` are associated with lower-frequency components that are treated as more relevant for downstream tasks.
- Different propagation depths apply powers `F^L` and `F^(L')`, which change the spectral emphasis of the two views.
- The theoretical analysis claims asymmetric depth encourages representations that better suppress high-frequency noise.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[gong-2022-magcl-2212-07035]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[gong-2022-magcl-2212-07035]].
