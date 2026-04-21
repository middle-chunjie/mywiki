---
type: concept
title: Tree-Based Negative Sampling
slug: tree-based-negative-sampling
date: 2026-04-20
updated: 2026-04-20
aliases: [树形负采样]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Tree-Based Negative Sampling** (树形负采样) — a structured negative-sampling strategy that uses sibling nodes in a retrieval tree as hard negatives for training node discrimination.

## Key Points

- In JTR, the positive at each level is the target leaf or one of its ancestors, while negatives are sampled from that positive node's siblings.
- The strategy exploits the fact that sibling nodes are usually close in embedding space because the tree is initialized by `k`-means.
- Sampling hard sibling negatives helps the learned node scores better support beam-search pruning.
- The method is one of the two key ingredients, together with unified contrastive loss, for joint optimization of the tree and query encoder.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[li-2023-constructing-2304-11943]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[li-2023-constructing-2304-11943]].
