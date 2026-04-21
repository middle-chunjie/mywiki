---
type: concept
title: Energy-Based Learning
slug: energy-based-learning
date: 2026-04-20
updated: 2026-04-20
aliases: [EBL, energy based learning, 能量学习]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Energy-Based Learning** (能量学习) — a learning framework that assigns lower energy to compatible input-output configurations and higher energy to incompatible ones, with training driven by relative energy differences.

## Key Points

- The paper shows that NT-Xent contrastive learning can be rewritten as an energy-based negative log-likelihood by defining `E(W, Y_i, X_i) = -sim(f(X_i), f(Y_i))`.
- This interpretation motivates adding an explicit margin objective over the hardest negative rather than relying only on the softmax normalization in contrastive loss.
- PromCSE uses the energy-based hinge loss `L_EH = [m + sim(h_i, h_i_hat) - sim(h_i, h_i^+)]_+` in supervised settings.
- With `lambda = 10` and best margin `m = 0.2`, the energy-based term consistently improves supervised SimCSE and PromCSE across BERT and RoBERTa backbones.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[jiang-2022-improved-2203-06875]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[jiang-2022-improved-2203-06875]].
