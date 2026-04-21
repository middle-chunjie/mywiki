---
type: concept
title: Cluster-aware Contrastive Learning
slug: cluster-aware-contrastive-learning
date: 2026-04-20
updated: 2026-04-20
aliases: [prototype-aware contrastive learning]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Cluster-aware Contrastive Learning** — a contrastive objective that supplements instance-level positive pairs with cluster-structured signals so semantically similar examples are aligned beyond exact identity matches.

## Key Points

- VGCL keeps the standard node-level InfoNCE objective for two sampled views of the same user or item.
- It adds a cluster-level loss in which positive weight is proportional to the probability that two nodes belong to the same K-means prototype.
- Separate prototype sets are learned for users and items, noted as `` `C^u \in R^{d \times K_u}` `` and `` `C^i \in R^{d \times K_i}` ``.
- The additional loss improves recommendation quality over plain node-level contrastive learning on all three datasets in the paper's ablations.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[yang-2023-generative-2307-05100]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[yang-2023-generative-2307-05100]].
