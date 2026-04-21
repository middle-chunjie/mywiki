---
type: concept
title: Policy Gradient
slug: policy-gradient
date: 2026-04-20
updated: 2026-04-20
aliases: []
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Policy gradient** (策略梯度) — a class of reinforcement-learning methods that updates a policy directly by following gradients of expected reward under trajectories sampled from that policy.

## Key Points

- Composer 2 uses a group-based policy-gradient setup with multiple rollouts per prompt, fixed group size, and single-epoch usage so prompts are not revisited.
- The report removes GRPO-style length standardization and avoids standard-deviation normalization of group advantages to reduce bias and degeneracy.
- For KL regularization, the paper defines `r = p(x) / q(x)` and prefers the estimator `k1 = -log r` over `k3 = (r - 1) - log r` because `k3` becomes high-variance when policies diverge.
- RL is fully asynchronous: rollout workers and training workers run independently, with weight synchronization and routing controls used to keep samples close to on-policy.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[research-2026-composer-2603-24477]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[research-2026-composer-2603-24477]].
