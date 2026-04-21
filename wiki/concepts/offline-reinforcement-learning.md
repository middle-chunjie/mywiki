---
type: concept
title: Offline Reinforcement Learning
slug: offline-reinforcement-learning
date: 2026-04-20
updated: 2026-04-20
aliases: [offline RL, batch reinforcement learning]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Offline Reinforcement Learning** (离线强化学习) — reinforcement learning from a fixed dataset of previously collected transitions without further interaction during training.

## Key Points

- [[dainese-2024-generating-2405-15383]] frames Code World Model synthesis in an offline RL setting with a fixed transition dataset `D`.
- In CWMB, most environments use only `10` trajectories, typically mixing random and higher-return behavior instead of requiring optimal demonstrations.
- The offline data is used mainly for validation and error feedback rather than gradient-based dynamics learning, which is a distinctive design choice.
- The paper argues that this setting can be much more sample efficient than standard offline RL methods because the code model is checked against trajectories instead of fit end-to-end from them.
- A preliminary comparison with CQL suggests the code-based approach is more favorable on some discrete tasks but less reliable on complex continuous environments.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[dainese-2024-generating-2405-15383]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[dainese-2024-generating-2405-15383]].
