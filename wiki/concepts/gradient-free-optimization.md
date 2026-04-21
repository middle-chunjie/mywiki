---
type: concept
title: Gradient-Free Optimization
slug: gradient-free-optimization
date: 2026-04-20
updated: 2026-04-20
aliases: [gradient free optimization, gradientless optimization, 无梯度优化]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Gradient-Free Optimization** (无梯度优化) — optimization that improves model parameters or prompts without requiring explicit analytic gradients from the underlying objective.

## Key Points

- The paper positions MeZO as a practical gradient-free optimizer for large language-model adaptation, rather than only for small prompt spaces or adversarial attacks.
- A key benefit in this work is compatibility with non-differentiable objectives such as direct accuracy or F1 optimization.
- The reported comparison against BBTv2 suggests that full-model or PEFT-level gradient-free adaptation can outperform lower-dimensional black-box tuning baselines.
- The study shows that gradient-free methods can become competitive with backpropagation when memory is the main bottleneck and pretrained models already provide favorable optimization geometry.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[malladi-2024-finetuning-2305-17333]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[malladi-2024-finetuning-2305-17333]].
