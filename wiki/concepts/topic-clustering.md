---
type: concept
title: Topic Clustering
slug: topic-clustering
date: 2026-04-20
updated: 2026-04-20
aliases: [topic discovery, 主题聚类]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Topic Clustering** (主题聚类) — the unsupervised grouping of text instances into semantically related clusters so that recurring themes in a corpus can be analyzed quantitatively.

## Key Points

- PRISM clusters opening prompts by embedding them with `all-mpnet-base-v2`, reducing them with UMAP to `20` dimensions, and clustering with HDBSCAN.
- With `min_cluster_size = 80`, about `70%` of prompts fall into `22` interpretable clusters while the remaining prompts are treated as outliers.
- Cluster labels are proposed from TF-IDF n-grams and centroid-near prompts, then manually checked to reduce labeling artifacts.
- The resulting clusters reveal both priming effects from task instructions and smaller but non-zero demographic differences in prompt selection.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[kirk-2024-prism-2404-16019]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[kirk-2024-prism-2404-16019]].
