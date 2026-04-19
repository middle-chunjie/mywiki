---
type: concept
title: Learning Rate Warmup
slug: learning-rate-warmup
date: 2026-04-17
updated: 2026-04-17
aliases: [Learning Rate Warmup, Warmup, Noam Schedule, 学习率预热]
tags: [optimization, training]
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-17
---

## Definition

Learning Rate Warmup (学习率预热) — an optimizer schedule that linearly increases the learning rate from near zero over an initial set of training steps before decaying it, used to stabilize training of deep attention-based networks.

## Key Points

- Transformer schedule (often called the "Noam schedule"): `lr = d_model^-0.5 · min(step^-0.5, step · warmup_steps^-1.5)`.
- For `step < warmup_steps`, the learning rate grows linearly; afterwards it decays as the inverse square root of the step count.
- [[vaswani-2017-attention-1706-03762]] uses `warmup_steps = 4000` with Adam (`β1 = 0.9, β2 = 0.98, ε = 1e-9`).
- Scales inversely with `√d_model`, so larger models automatically receive smaller peak learning rates.
- Widely inherited by later Transformer training pipelines, with variations in warmup length and decay curve.

## My Position

<!-- User's stance on this concept. Fed by personal writing; tag "(personal stance)" on such bullets. -->

## Contradictions

<!-- Disagreements among sources about this concept. -->

## Sources

- [[vaswani-2017-attention-1706-03762]]

## Evolution Log

- 2026-04-17 (1 source): established from [[vaswani-2017-attention-1706-03762]] with `warmup_steps = 4000`, inverse-sqrt decay, paired with Adam.
