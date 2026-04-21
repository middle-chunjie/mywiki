---
type: concept
title: Minimax Optimization
slug: minimax-optimization
date: 2026-04-20
updated: 2026-04-20
aliases: [minimax game, 极小化极大优化]
tags: [optimization, game-theory, adversarial-training]
source_count: 1
confidence: low
domain_volatility: low
last_reviewed: 2026-04-20
---

## Definition

**Minimax Optimization** (极小化极大优化) — a two-player zero-sum game objective of the form `min_θ max_φ J(θ, φ)`, where one player minimizes and the other maximizes the same objective, driving both toward a Nash equilibrium.

## Key Points

- Foundational to GANs (Goodfellow et al., 2014): the generator minimizes while the discriminator maximizes a divergence measure between real and generated data distributions.
- In AR2, the contrastive minimax objective is: `min_θ max_φ E_{D_q^- ~ G_θ} [log p_φ(d | q; D_q)]`. The retriever (θ) minimizes by trying to fool the ranker; the ranker (φ) maximizes by correctly identifying the ground-truth document.
- Practical optimization uses alternating updates: fix one player and optimize the other, then swap — a common approximation to true simultaneous minimax.
- Training instability is a known issue; distillation regularization in AR2 smooths the retriever distribution and helps convergence.
- The minimax framing provides a principled justification for progressive hard-negative generation: the retriever is incentivized to surface genuinely confusing candidates rather than arbitrary hard negatives.

## My Position

<!-- User's stance on this concept. Fed by personal writing; tag "(personal stance)" on such bullets. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[zhang-2022-adversarial]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[zhang-2022-adversarial]].
