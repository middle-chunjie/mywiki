---
type: concept
title: Compute-Optimal Test-Time Scaling
slug: compute-optimal-test-time-scaling
date: 2026-04-20
updated: 2026-04-20
aliases: [compute-optimal inference scaling, adaptive test-time scaling, 测试时计算最优扩展]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Compute-Optimal Test-Time Scaling** (测试时计算最优扩展) — a prompt-dependent policy that allocates a fixed inference budget to the test-time strategy expected to maximize accuracy on that prompt.

## Key Points

- The paper formalizes the target policy as choosing `theta` that maximizes expected correctness for a given prompt `q` and budget `N`.
- In practice, the approximation groups prompts into five difficulty bins and chooses the best strategy per bin.
- The optimal allocation differs across prompts: easy cases favor sequential revision, while harder ones often benefit from more search.
- The adaptive policy improves efficiency by roughly `2x-4x` over fixed best-of-`N` style baselines.
- The same idea is used for both verifier search and revision-based proposal refinement.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[snell-2024-scaling-2408-03314]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[snell-2024-scaling-2408-03314]].
