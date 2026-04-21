---
type: concept
title: Group Relative Policy Optimization
slug: group-relative-policy-optimization
date: 2026-04-20
updated: 2026-04-20
aliases: [GRPO]
tags: [llm, optimization]
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Group Relative Policy Optimization** — a reinforcement-learning algorithm used to optimize language models from sampled rollouts with reward signals defined over generated outputs.

## Key Points

- The paper instantiates LCPO with GRPO, while noting the method is not specific to this optimizer.
- GRPO is used for both the exact-length and maximum-budget training stages of L1.
- The reported training recipe uses `lr = 1e-6`, batch size `128`, and `700` RL steps for the first stage.
- A second `120`-step GRPO phase adapts L1-Exact into L1-Max with the max-budget objective.
- The need for smooth gradients partly motivates the clipped soft-constraint design used in the L1-Max reward.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[aggarwal-2025-l-2503-04697]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[aggarwal-2025-l-2503-04697]].
