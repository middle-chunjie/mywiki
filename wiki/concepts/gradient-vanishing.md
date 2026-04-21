---
type: concept
title: Gradient Vanishing
slug: gradient-vanishing
date: 2026-04-20
updated: 2026-04-20
aliases: [Vanishing Gradient, Vanishing Gradient Problem, 梯度消失]
tags: [deep-learning, optimization, training-pathology]
source_count: 1
confidence: low
domain_volatility: low
last_reviewed: 2026-04-20
---

## Definition

**Gradient Vanishing** (梯度消失) — a training pathology in deep networks where gradients become exponentially small as they are backpropagated through many layers, preventing early layers from learning effectively.

## Key Points

- Originally identified in recurrent networks by Bengio et al. (1994); later recognized as a fundamental obstacle in deep feed-forward architectures.
- [[residual-connection]] was a major advance against gradient vanishing: the additive identity shortcut creates gradient highways that bypass the multiplicative attenuation of successive layer Jacobians.
- [[post-norm]] reintroduces vanishing gradients by normalizing after the residual addition, attenuating the shortcut signal before it can protect gradient flow.
- [[pre-norm]] largely resolves gradient vanishing by normalizing the transformation branch before it enters the main residual path.
- In the [[hyper-connections]] framework, the seesaw between gradient vanishing and [[representation-collapse]] motivates learning connection strengths rather than fixing them via normalization placement.
- Practically, gradient vanishing manifests as near-zero parameter updates in early layers, slow learning, and training instability in deep networks.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[zhu-2025-hyperconnections-2409-19606]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[zhu-2025-hyperconnections-2409-19606]] as one of the two central failure modes motivating hyper-connections.
