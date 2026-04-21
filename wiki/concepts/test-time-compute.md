---
type: concept
title: Test-Time Compute
slug: test-time-compute
date: 2026-04-20
updated: 2026-04-20
aliases: [inference-time compute, test time compute, 测试时计算]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Test-Time Compute** (测试时计算) — additional computation spent during inference to improve a model's output beyond a single greedy or sampled pass.

## Key Points

- The paper studies test-time compute as a separate scaling axis from model parameter count.
- It analyzes two main uses of extra inference compute: verifier-guided search and sequential answer revision.
- The value of more test-time compute depends strongly on prompt difficulty rather than following a uniform scaling rule.
- On easier questions, extra compute often works best as iterative refinement; on harder questions, broader search is more useful.
- In FLOPs-matched settings, extra inference compute can sometimes outperform scaling the base model's parameters.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[snell-2024-scaling-2408-03314]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[snell-2024-scaling-2408-03314]].
