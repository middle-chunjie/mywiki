---
type: concept
title: Efficiency-Aware Policy Optimization
slug: efficiency-aware-policy-optimization
date: 2026-04-20
updated: 2026-04-20
aliases: [EAPO]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Efficiency-Aware Policy Optimization** — a reinforcement-learning objective that biases an agent toward correct but shorter trajectories by discounting rewards according to how long success takes.

## Key Points

- EAPO assigns each round reward `r_t = γ^(T-t) · R_T`, so earlier completion yields larger credit.
- The method is built on top of GSPO while normalizing advantages across all rounds in the same question group.
- Because each rollout contributes one sample per round, EAPO trains on a denser corpus than trajectory-level RL.
- Adaptive downsampling keeps the number of retained samples divisible by data-parallel size with minimal data loss.
- In the paper's ablation, EAPO slightly improves average score over GSPO while reducing average interaction count.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[chen-2025-iterresearch-2511-07327]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[chen-2025-iterresearch-2511-07327]].
