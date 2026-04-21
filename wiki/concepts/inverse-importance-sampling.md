---
type: concept
title: Inverse Importance Sampling
slug: inverse-importance-sampling
date: 2026-04-20
updated: 2026-04-20
aliases: [inverse importance sampling, 逆重要性采样]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Inverse Importance Sampling** (逆重要性采样) - a sampling strategy that upweights low-importance items by drawing them with probability inversely proportional to a predefined importance score.

## Key Points

- RAGraph defines node importance as `I(v) = alpha * PR(v) + (1 - alpha) * DC(v)` by mixing PageRank and degree centrality.
- The sampling weight is reversed to `I'(v) = 1 / (I(v) + epsilon)`, so lower-importance nodes become more likely master nodes for toy graphs.
- Normalized probabilities `p_i` are then used in weighted sampling over each resource-graph snapshot.
- The paper motivates this design as a way to surface long-tail graph knowledge that standard pre-training may underrepresent.
- The same reversed-importance signal also controls how many augmented toy graphs are generated from each ego net.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[jiang-2024-ragraph-2410-23855]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[jiang-2024-ragraph-2410-23855]].
