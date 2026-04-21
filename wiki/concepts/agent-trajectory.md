---
type: concept
title: Agent Trajectory
slug: agent-trajectory
date: 2026-04-20
updated: 2026-04-20
aliases: [interaction trajectory, rollout trace]
tags: [agents, training]
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Agent Trajectory** (智能体轨迹) — the ordered interaction history of an agent with its environment, typically including observations, actions, intermediate outputs, and resulting feedback.

## Key Points

- SWE-Gym provides executable feedback that lets the authors label trajectories as successful or unsuccessful.
- The OpenHands verifier models a trajectory as `tau = [o_1, a_1, ..., o_n, a_n]`.
- Successful trajectories are used for rejection-sampling fine-tuning, while both successes and failures are used for verifier training.
- The paper reports that successful OpenHands trajectories average about 39.9 messages and 18.6k tokens.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[pan-2024-training-2412-21139]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[pan-2024-training-2412-21139]].
