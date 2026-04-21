---
type: concept
title: Goal-Directed Exploration
slug: goal-directed-exploration
date: 2026-04-20
updated: 2026-04-20
aliases: [Purposeful exploration, 目标导向探索]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Goal-Directed Exploration** (目标导向探索) — an exploration strategy that preferentially visits states predicted to advance the agent toward potential reward rather than sampling actions uniformly at random.

## Key Points

- In WorldCoder, goal-directed exploration emerges from enforcing `\phi_2`, which requires the current world model to support a plan to positive reward.
- Before the agent has observed true reward, the reward model may invent plausible intermediate objectives that lie within the agent's current zone of proximal development.
- This mechanism is most important in sparse-reward settings such as MiniGrid UnlockPickup and AlfWorld, where random exploration is ineffective.
- The paper contrasts this with dense-reward Sokoban, where optimism has relatively little effect because random exploration already reveals useful supervision.
- The method ties exploration to symbolic planning, letting the planner probe states that are textually or structurally aligned with the current goal.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[tang-2024-worldcoder-2402-12275]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[tang-2024-worldcoder-2402-12275]].
