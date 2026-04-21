---
type: concept
title: Cross-Entropy Method
slug: cross-entropy-method
date: 2026-04-20
updated: 2026-04-20
aliases: [CEM, Cross Entropy Method]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Cross-Entropy Method** (交叉熵方法) — a stochastic optimization procedure that iteratively fits a sampling distribution to high-performing candidates and resamples from the updated distribution.

## Key Points

- [[dainese-2024-generating-2405-15383]] uses CEM as the planner for continuous-action environments once a Code World Model has been synthesized.
- The planner optimizes action sequences over horizon `T_cem = 100` by sampling `N_cem = 1000` candidate plans per iteration and refitting to the top `K_cem = 100` elites.
- The paper runs `I_cem = 20` refinement iterations and clips sampled plans to the legal action bounds of each environment.
- CEM is part of the downstream evaluation stack rather than the code-generation loop itself, serving to test whether a learned CWM is useful for actual control.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[dainese-2024-generating-2405-15383]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[dainese-2024-generating-2405-15383]].
