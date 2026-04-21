---
type: concept
title: Mixed-Integer Programming
slug: mixed-integer-programming
date: 2026-04-20
updated: 2026-04-20
aliases: [MIP, mixed integer programming, 混合整数规划]
tags: []
source_count: 1
confidence: low
domain_volatility: low
last_reviewed: 2026-04-20
---

## Definition

**Mixed-Integer Programming** (混合整数规划) — an optimization framework that solves problems with both discrete decision variables and continuous or logical constraints.

## Key Points

- The paper formulates robust `k`-center feasibility as an MIP parameterized by a candidate covering radius `δ`.
- Binary variables indicate which points are selected as centers, which centers cover which points, and which assignments are treated as outliers.
- The MIP is embedded in a binary search over `δ` to refine the greedy `2`-approximate solution.
- The formulation explicitly supports an outlier budget `Ξ`, making the selection robust to a small number of uncovered points.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[sener-2018-active-1708-00489]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[sener-2018-active-1708-00489]].
