---
type: concept
title: Monte Carlo Sampling
slug: monte-carlo-sampling
date: 2026-04-20
updated: 2026-04-20
aliases: [MC sampling]
tags: [sampling, optimization]
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Monte Carlo Sampling** — an estimation procedure that approximates a quantity of interest by averaging outcomes over repeated stochastic samples or rollouts.

## Key Points

- In [[xiong-2025-mpo-2503-02682]], Monte Carlo sampling estimates meta-plan quality by averaging task rewards over `N` sampled execution trajectories.
- The same paper uses sampled plan sets of size `M` to build preference pairs for DPO by selecting the highest- and lowest-quality plans for each task.
- The reported sensitivity analysis shows that plan quality deteriorates when `N` or `M` is too small, with rollout count `N` affecting quality more strongly.
- MPO chooses `N = M = 5` as a practical tradeoff between estimator quality and sampling cost.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[xiong-2025-mpo-2503-02682]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[xiong-2025-mpo-2503-02682]].
