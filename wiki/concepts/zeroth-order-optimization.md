---
type: concept
title: Zeroth-Order Optimization
slug: zeroth-order-optimization
date: 2026-04-20
updated: 2026-04-20
aliases: [ZO, zeroth order optimization, derivative-free optimization, 零阶优化]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Zeroth-Order Optimization** (零阶优化) — an optimization family that updates model parameters using only function or loss evaluations, without explicit backpropagated gradients.

## Key Points

- The paper instantiates zeroth-order optimization for language-model fine-tuning through SPSA-based projected gradient estimates computed from only two forward passes.
- Its central engineering claim is that a seed-replay implementation can make ZO memory usage essentially match inference-time memory, removing the usual backward-pass activation burden.
- MeZO shows that ZO can remain practically useful even for models up to `66B` parameters, contrary to the standard intuition that high-dimensional ZO is catastrophically slow.
- The reported empirical tradeoff is favorable memory with worse step efficiency in iteration count: MeZO often needs many more steps than full fine-tuning but far fewer GPUs and much less memory.
- The theory section argues that under favorable local geometry, ZO slowdown depends on effective rank rather than raw parameter dimension.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[malladi-2024-finetuning-2305-17333]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[malladi-2024-finetuning-2305-17333]].
