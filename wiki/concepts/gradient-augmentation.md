---
type: concept
title: Gradient Augmentation
slug: gradient-augmentation
date: 2026-04-20
updated: 2026-04-20
aliases: [Gradient Diversification, 梯度增强]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Gradient Augmentation** (梯度增强) — a training procedure that perturbs or enriches gradient signals, here by adding losses from sampled subnetworks, to encourage more diverse and robust learned representations.

## Key Points

- RoPGen samples `n` subnetworks with widths `` `w_j ∈ [α, 1]` `` during each training iteration.
- The subnetwork loss is summed with the full-network loss so that `` `L_RoPGen = L_std + L_subnet` ``.
- The paper interprets the resulting gradient as `` `g_RoPGen = g_std + g_subnet` ``, where `g_subnet` augments the standard gradient.
- Ablation results show that removing gradient augmentation weakens robustness against both adversarial-example and style-based attacks.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[li-2022-ropgen-2202-06043]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[li-2022-ropgen-2202-06043]].
