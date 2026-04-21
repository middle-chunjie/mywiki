---
type: concept
title: K-Center Problem
slug: k-center-problem
date: 2026-04-20
updated: 2026-04-20
aliases: [k-center, k-center clustering, k-center problem]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**K-Center Problem** — a combinatorial optimization problem that chooses `k` centers to minimize the maximum distance from any point to its nearest selected center.

## Key Points

- The paper shows that minimizing the active-learning covering radius is equivalent to a `k`-center objective in CNN feature space.
- The distance metric is the `l_2` distance between activations of the final fully connected layer.
- A greedy farthest-first traversal provides a `2`-approximation that is already strong enough to outperform competing baselines.
- The authors further refine the greedy solution with a feasibility-based mixed-integer program that can ignore a bounded number of outliers.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[sener-2018-active-1708-00489]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[sener-2018-active-1708-00489]].
