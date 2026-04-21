---
type: concept
title: Randomized Smoothing
slug: randomized-smoothing
date: 2026-04-20
updated: 2026-04-20
aliases: [RS, 随机平滑]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Randomized Smoothing** (随机平滑) — a technique that optimizes an expectation over randomly perturbed inputs or variables to smooth a rugged objective landscape.

## Key Points

- The paper defines a smoothed loss `l_smooth(z,u) = E[l_attack(z + μξ, u + μτ)]` to make adversarial program optimization easier.
- Noise samples are drawn from the unit Euclidean ball, with smoothing parameter `μ = 0.01`.
- The Monte Carlo approximation uses `m = 10` samples in practice.
- The strongest gains come from smoothing the perturbation variables `u`, and AO+RS delivers the best ASR across both Python and Java experiments.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[srikant-2021-generating-2103-11882]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[srikant-2021-generating-2103-11882]].
