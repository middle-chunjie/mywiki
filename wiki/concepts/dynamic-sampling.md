---
type: concept
title: Dynamic Sampling
slug: dynamic-sampling
date: 2026-04-20
updated: 2026-04-20
aliases: [adaptive batch resampling, 动态采样]
tags: [reinforcement-learning, reasoning]
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Dynamic Sampling** (动态采样) — a reinforcement learning batching strategy that keeps sampling until each retained prompt yields non-degenerate reward variation, so the batch preserves effective policy-gradient signal.

## Key Points

- In DAPO, prompts whose sampled responses are all correct or all incorrect are filtered because their group-normalized advantages collapse to zero.
- The retained batch satisfies `0 < |{o_i | is_equivalent(a, o_i)}| < G`, which preserves non-zero gradients for every prompt used in training.
- The paper motivates the method as a fix for shrinking effective batch size during long-CoT RL.
- Although dynamic sampling increases the amount of generated data, the paper reports faster convergence in wall-clock training because fewer update steps are needed.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[yu-2025-dapo-2503-14476]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[yu-2025-dapo-2503-14476]].
