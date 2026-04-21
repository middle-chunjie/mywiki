---
type: concept
title: Distillation Regularization
slug: distillation-regularization
date: 2026-04-20
updated: 2026-04-20
aliases: [regularization by distillation, 蒸馏正则化]
tags: [knowledge-distillation, regularization, training, dense-retrieval]
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Distillation Regularization** (蒸馏正则化) — a training technique that adds a cross-entropy loss between a teacher model's output distribution and a student model's distribution as a regularization term, preventing the student from overfitting to sharp predictions while transferring the teacher's generalization.

## Key Points

- In AR2, the regularization term is `J_R^θ = H(p_φ(·|q; D_q), p_θ(·|q; D_q))`, where the cross-encoder ranker (`φ`) acts as teacher and the dual-encoder retriever (`θ`) is the student.
- Without regularization, the retriever distribution collapses to near-zero entropy (entropy 1.70 vs. 2.10 with regularization on NQ test set), reflecting over-confident retrieval probabilities that impede exploration during training.
- The distillation loss acts as a smooth correction term on top of the adversarial objective, analogous to entropy regularization in reinforcement learning.
- Regularization improves NQ R@1 from 57.8 → 58.7 and R@5 from 77.3 → 77.9 in AR2 ablation.
- Distinct from full knowledge distillation: only the output distribution over a small candidate set `D_q` is distilled, not the full corpus; the ranker is not a frozen teacher but jointly trained alongside the retriever.

## My Position

<!-- User's stance on this concept. Fed by personal writing; tag "(personal stance)" on such bullets. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[zhang-2022-adversarial]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[zhang-2022-adversarial]].
