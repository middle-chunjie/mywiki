---
type: concept
title: Constrained K-Means Clustering
slug: constrained-k-means-clustering
date: 2026-04-20
updated: 2026-04-20
aliases: [constrained k-means, 约束 K 均值聚类]
tags: [clustering, discrete-representation]
source_count: 1
confidence: low
domain_volatility: low
last_reviewed: 2026-04-20
---

## Definition

**Constrained K-Means Clustering** (约束 K 均值聚类) — a clustering variant that imposes assignment constraints so clusters remain more balanced than in unconstrained `k`-means.

## Key Points

- [[sun-2023-tokenize-2304-04171]] uses constrained `K`-means to initialize each timestep codebook with more balanced centroids.
- The goal is to avoid collapsed docid usage where only a few codes receive most documents early in training.
- The paper formulates the constrained assignment step as a minimum-cost-flow problem to improve codebook diversity.
- Better balanced codebooks shorten docids and improve online retrieval efficiency by reducing downstream conflicts.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[sun-2023-tokenize-2304-04171]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[sun-2023-tokenize-2304-04171]].
