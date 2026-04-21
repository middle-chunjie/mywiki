---
type: concept
title: Semantic Clustering
slug: semantic-clustering
date: 2026-04-20
updated: 2026-04-20
aliases: [embedding-based clustering, 语义聚类]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Semantic Clustering** (语义聚类) — grouping text or code artifacts by similarity in an embedding space so that semantically related items fall into the same cluster.

## Key Points

- CodeChain embeds extracted code sub-modules and applies `K`-means to cluster related implementations across sampled programs.
- Cluster centroids are used to select representative reusable sub-modules rather than sampling whole programs directly.
- The paper finds clustering is a better revision signal than random selection, but it works best after filtering out obviously bad programs.
- A decreasing cluster schedule is reported as the best setting because it moves from early exploration to later exploitation across revision rounds.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[unknown-nd-codechaintowards-2310-08992]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[unknown-nd-codechaintowards-2310-08992]].
