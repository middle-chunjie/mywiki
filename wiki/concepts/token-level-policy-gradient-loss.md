---
type: concept
title: Token-Level Policy Gradient Loss
slug: token-level-policy-gradient-loss
date: 2026-04-20
updated: 2026-04-20
aliases: [token-level loss, 词元级策略梯度损失]
tags: [reinforcement-learning, optimization]
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Token-Level Policy Gradient Loss** (词元级策略梯度损失) — a policy-gradient reduction scheme that averages RL loss over all generated tokens rather than first averaging within each sampled sequence.

## Key Points

- DAPO replaces sample-level GRPO reduction with normalization by the total token count `Σ_i |o_i|`.
- The change is motivated by long chain-of-thought settings, where equal sample weighting under-emphasizes informative tokens inside long responses.
- The paper argues that token-level reduction both reinforces useful long reasoning traces and suppresses low-quality overlong patterns such as repetition and gibberish.
- In the ablation sequence, adding token-level loss improves AIME 2024 avg@32 from `41` to `42` and is reported to improve training stability.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[yu-2025-dapo-2503-14476]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[yu-2025-dapo-2503-14476]].
