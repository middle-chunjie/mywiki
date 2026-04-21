---
type: concept
title: Datastore Pruning
slug: datastore-pruning
date: 2026-04-20
updated: 2026-04-20
aliases: [memory pruning, datastore compression, 数据存储库剪枝]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Datastore Pruning** (数据存储库剪枝) — reducing the number of external memory entries used by a retrieval model while trying to preserve retrieval quality.

## Key Points

- The paper studies pruning as one of three main axes for accelerating `kNN-LM` inference.
- Four pruning strategies are compared: random pruning, target-aware `k`-means, rank-based pruning, and greedy merging.
- When multiple original entries are collapsed, the corrected retrieval distribution weights each compressed entry by its multiplicity `s_i`.
- Results show that pruning quality depends on preserving local token-consistent neighborhoods rather than only global compression ratio.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[he-2021-efficient-2109-04212]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[he-2021-efficient-2109-04212]].
