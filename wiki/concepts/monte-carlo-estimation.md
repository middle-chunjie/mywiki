---
type: concept
title: Monte Carlo Estimation
slug: monte-carlo-estimation
date: 2026-04-20
updated: 2026-04-20
aliases: [MC estimation, Monte Carlo estimation, 蒙特卡罗估计]
tags: [search, verification]
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Monte Carlo Estimation** (蒙特卡罗估计) — a method that estimates an unknown quantity by sampling multiple stochastic outcomes and computing their empirical ratio or average.

## Key Points

- OmegaPRM estimates prefix correctness by sampling multiple completions from a partial chain-of-thought and measuring the fraction that reach the correct final answer.
- The paper uses this value as `` `MC(s)` `` for each tree state and treats `` `MC(s) > 0` `` as evidence that the current prefix can still be completed correctly.
- Each estimate uses `` `k = 8` `` rollouts in the main data-generation setup.
- These Monte Carlo estimates drive both binary-search error localization and rollout prioritization inside the search tree.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[luo-2024-improve-2406-06592]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[luo-2024-improve-2406-06592]].
