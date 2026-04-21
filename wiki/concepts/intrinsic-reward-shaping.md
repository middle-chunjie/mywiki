---
type: concept
title: Intrinsic Reward Shaping
slug: intrinsic-reward-shaping
date: 2026-04-20
updated: 2026-04-20
aliases: [内在奖励塑形]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Intrinsic Reward Shaping** (内在奖励塑形) — the addition of auxiliary reward terms that bias learning toward desired behaviors beyond the environment's original reward.

## Key Points

- D2Skill adds a hindsight intrinsic reward `R_i^{int} = \lambda (Y_i - \bar{Y}_g^{base})` to skill-injected rollouts.
- The shaping reward directly measures whether retrieved skills improved outcome over the matched baseline group.
- This reward is only applied to skill trajectories, letting the policy learn not just from success but from relative benefit due to skill usage.
- The paper positions this mechanism as a bridge between skill valuation and policy optimization.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[tu-2026-dynamic-2603-28716]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[tu-2026-dynamic-2603-28716]].
