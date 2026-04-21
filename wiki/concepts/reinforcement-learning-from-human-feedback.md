---
type: concept
title: Reinforcement Learning from Human Feedback
slug: reinforcement-learning-from-human-feedback
date: 2026-04-20
updated: 2026-04-20
aliases: [RLHF, 人类反馈强化学习]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Reinforcement Learning from Human Feedback** (人类反馈强化学习) — an alignment pipeline that trains a model from preference signals, typically by fitting a reward model and optimizing the policy to prefer higher-reward outputs.

## Key Points

- The paper treats PPO-based RLHF as the dominant training-based baseline for LLM alignment and contrasts it with input-side prompt optimization.
- RLHF requires white-box access, extra optimization stages, and more compute, making it unusable for many API-only systems.
- On Vicuna, BPO is competitive with PPO and can be composed with PPO to yield additional gains instead of being purely a substitute.
- The paper frames BPO as complementary to RLHF: one changes prompts, the other changes model parameters.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[cheng-2024-blackbox-2311-04155]]
- [[menick-2022-teaching-2203-11147]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[cheng-2024-blackbox-2311-04155]].
