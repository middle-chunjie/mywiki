---
type: concept
title: Trustworthy Process Rewarding
slug: trustworthy-process-rewarding
date: 2026-04-20
updated: 2026-04-20
aliases: [可信过程奖励]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Trustworthy Process Rewarding** (可信过程奖励) — a step-level feedback mechanism that combines reliable reward scoring and explanatory critiques so a reasoning system can search, refine, and train on multi-step trajectories more safely.

## Key Points

- [[sun-2025-rearter-2501-07861]] implements the mechanism with a paired [[process-reward-model]] and [[process-explanation-model]] rather than a scalar verifier alone.
- The paper treats trustworthiness as a data-and-inference issue: PRM training data is rebalanced with OmegaPRM-style annotation and stronger generators for hard examples.
- Early-step bias is reduced with a TD-style look-ahead update `r_t <- r_t + \alpha (r_{t+1} - r_t)` plus adaptive stopping.
- The same reward mechanism is reused in both test-time search and post-training data collection, tying inference-time critique to offline optimization.
- The paper argues that trustworthy process rewards produce better reasoning paths than PRM-only planning baselines such as CR-Planner.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[sun-2025-rearter-2501-07861]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[sun-2025-rearter-2501-07861]].
