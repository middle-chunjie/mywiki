---
type: concept
title: K-Means Clustering
slug: k-means-clustering
date: 2026-04-20
updated: 2026-04-20
aliases: [kmeans, k-means, K 均值聚类]
tags: [clustering, indexing]
source_count: 1
confidence: low
domain_volatility: low
last_reviewed: 2026-04-20
---

## Definition

**K-Means Clustering** (K 均值聚类) — a centroid-based clustering algorithm that partitions points into `k` groups by minimizing within-cluster squared distance.

## Key Points

- The paper uses hierarchical `k`-means as both a baseline indexer (HKmI) and the core optimizer inside BMI.
- Under Gaussian assumptions for `p(Q|X=x)` and `p(Q|T=t)`, `k`-means is shown to maximize the BMI likelihood.
- BMI differs from HKmI only in what gets clustered: query-derived means `\mu_{Q|x}` instead of document embeddings `\mu_x`.
- The hierarchy uses alphabet `V = [1,\ldots,30]` and depth `m` chosen so that `|V|^m \ge |\mathcal{X}|`.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[du-2024-bottleneckminimal-2405-10974]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[du-2024-bottleneckminimal-2405-10974]].
