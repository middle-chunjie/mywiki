---
type: concept
title: On-Policy Reinforcement Learning
slug: on-policy-reinforcement-learning
date: 2026-04-20
updated: 2026-04-20
aliases: [same-policy RL, 同策略强化学习]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**On-Policy Reinforcement Learning** (同策略强化学习) — reinforcement learning in which optimization uses trajectories sampled from the current behavior policy rather than relying on fixed offline data.

## Key Points

- AgentFlow collects rollouts from the current planner inside the active multi-turn system before updating the planner.
- The paper argues this avoids the distribution shift that arises when training on curated traces outside the live interaction loop.
- The objective optimizes expected return over planner-generated trajectories while keeping the policy close to a reference model through KL regularization.
- Only the planner is updated on-policy; the other AgentFlow modules remain fixed during training.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[li-2026-intheflow-2510-05592]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[li-2026-intheflow-2510-05592]].
