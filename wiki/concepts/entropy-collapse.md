---
type: concept
title: Entropy Collapse
slug: entropy-collapse
date: 2026-04-20
updated: 2026-04-20
aliases: [policy entropy collapse, 熵塌缩]
tags: [reinforcement-learning, failure-mode]
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Entropy Collapse** (熵塌缩) — a failure mode in policy optimization where the model's output distribution becomes prematurely sharp, reducing exploration and causing sampled responses to become nearly identical.

## Key Points

- DAPO identifies entropy collapse as a central failure mode of naive PPO and GRPO in long-CoT RL.
- The paper links the problem to overly restrictive upper clipping, which makes it hard for low-probability exploratory tokens to gain probability mass.
- Figure 2 reports that entropy decreases rapidly under the baseline and remains healthier after Clip-Higher is introduced.
- The authors treat entropy, mean generation probability, reward, and response length as core monitoring signals during training.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[yu-2025-dapo-2503-14476]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[yu-2025-dapo-2503-14476]].
