---
type: concept
title: Hindsight Optimization
slug: hindsight-optimization
date: 2026-04-20
updated: 2026-04-20
aliases: [事后优化]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Hindsight Optimization** (事后优化) — an optimization scheme that uses post-rollout comparisons or retrospective signals to revalue actions, skills, or trajectories after their outcomes are observed.

## Key Points

- D2Skill compares skill-injected and baseline rollout groups from the same policy to form hindsight task-level and trajectory-level utility signals.
- The task-level signal is the success-rate gap between the two groups, while the step-level signal is each trajectory's gain over the baseline-group mean.
- These retrospective signals update skill utilities and also define an intrinsic reward used in policy optimization.
- The method ties skill valuation directly to observed marginal benefit rather than to heuristic frequency or standalone language quality.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[tu-2026-dynamic-2603-28716]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[tu-2026-dynamic-2603-28716]].
