---
type: concept
title: Proximal Policy Optimization
slug: proximal-policy-optimization
date: 2026-04-20
updated: 2026-04-20
aliases: [PPO, PPO-Clip, 近端策略优化]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Proximal Policy Optimization** (近端策略优化) — a reinforcement learning algorithm that stabilizes policy updates by clipping the policy ratio during optimization.

## Key Points

- BIDER uses clipped PPO in the preference-alignment stage to optimize the evidence refiner at the token level.
- The paper's PPO objective combines clipped policy loss, value-function loss, and an entropy bonus.
- Rewards are sparse and sequence-level: they are issued only at `EOF`, based on the answer-quality delta between refined and original evidence.
- In this paper, PPO is not used to train the generator directly; it is used to reshape the intermediate evidence presented to the generator.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[jin-2024-bider-2402-12174]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[jin-2024-bider-2402-12174]].
