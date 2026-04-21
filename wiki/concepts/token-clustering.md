---
type: concept
title: Token Clustering
slug: token-clustering
date: 2026-04-20
updated: 2026-04-20
aliases: [term clustering, 词元聚类]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Token Clustering** (词元聚类) — the grouping of contextualized token representations into centroid-based partitions so that retrieval or compression can operate on clusters instead of full token vectors.

## Key Points

- PLAID relies on ColBERTv2 token centroids as coarse proxies for exact token representations during candidate generation.
- Document token vectors are represented as a centroid plus a quantized residual, reducing storage by about one order of magnitude relative to original ColBERT.
- The paper's analysis finds that many clusters are dominated by a single lexical token, with median majority-token proportion `0.86`.
- Tokens are not perfectly one-cluster-one-token: the median majority-cluster proportion is only `0.62`, so some words spread over multiple clusters.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[macavaney-2024-reproducibility]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[macavaney-2024-reproducibility]].
