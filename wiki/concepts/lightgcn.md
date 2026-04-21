---
type: concept
title: LightGCN
slug: lightgcn
date: 2026-04-20
updated: 2026-04-20
aliases: [Light Graph Convolution Network, 轻量图卷积网络]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**LightGCN** (轻量图卷积网络) — a graph collaborative filtering model that keeps only normalized neighborhood propagation and layer-wise embedding aggregation, removing feature transforms and nonlinearities.

## Key Points

- [[jin-2023-code]] adopts the LightGCN propagation rule for user-file aggregation rather than a heavier GCN block.
- The propagation uses symmetric normalization `1 / \sqrt{|\mathcal{N}_i||\mathcal{N}_j|}` and mean pooling across `L = 4` layers.
- LightGCN is also a strong baseline in the experiments and remains close to CODER in inference speed.
- CODER extends the LightGCN-style interaction modeling with code semantics, project hierarchy, and project-level behaviors.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[jin-2023-code]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[jin-2023-code]].
