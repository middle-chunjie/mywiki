---
type: concept
title: Over-smoothing
slug: over-smoothing
date: 2026-04-20
updated: 2026-04-20
aliases: [过平滑, oversmoothing]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Over-smoothing** (过平滑) — the degeneration in deep graph propagation where node embeddings become too similar, reducing discriminability for downstream tasks.

## Key Points

- The paper identifies over-smoothing as a risk in multi-layer GNN recommendation models.
- Residual propagation is used in the base encoder to reduce the collapse of node representations across layers.
- DCCF further argues that cross-view contrastive learning improves representation uniformity and helps counter over-smoothing.
- The evaluation uses MAD to quantify the effect, with higher MAD for DCCF than DCCF-CL on both Amazon-book and Tmall user embeddings.
- Compared baselines such as DGCL, DisenGCN, and LightGCN show lower MAD than DCCF in most reported settings.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[ren-2023-disentangled]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[ren-2023-disentangled]].
