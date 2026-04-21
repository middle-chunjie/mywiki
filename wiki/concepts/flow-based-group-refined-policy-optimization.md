---
type: concept
title: Flow-Based Group Refined Policy Optimization
slug: flow-based-group-refined-policy-optimization
date: 2026-04-20
updated: 2026-04-20
aliases: [Flow-GRPO]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Flow-Based Group Refined Policy Optimization** — an on-policy optimization method for agentic systems that trains a planner from live multi-turn rollouts by broadcasting one final outcome reward to every turn and normalizing advantages across a rollout group.

## Key Points

- The algorithm operates on full in-the-flow rollouts generated inside AgentFlow rather than on offline traces.
- Every planner action in a trajectory receives the same final-outcome reward, turning long-horizon optimization into a sequence of tractable single-turn updates.
- Flow-GRPO uses PPO-style clipping, token-level importance ratios, and a KL penalty to a reference policy.
- The paper further stabilizes training with group-normalized advantages computed across `G = 8` sampled rollouts per example.
- Empirically, Flow-GRPO raises the planner's average score from `38.5` to `55.7` in the planner-training ablation.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[li-2026-intheflow-2510-05592]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[li-2026-intheflow-2510-05592]].
