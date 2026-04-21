---
type: concept
title: Adversarial Robustness
slug: adversarial-robustness
date: 2026-04-20
updated: 2026-04-20
aliases: [model robustness to adversarial examples, adversarial resistance, 对抗鲁棒性]
tags: [robustness, adversarial, deep-learning, evaluation]
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Adversarial Robustness** (对抗鲁棒性) — the property of a model to maintain stable, correct outputs when presented with adversarially perturbed inputs that are designed to induce errors while remaining imperceptible or functionally equivalent to benign inputs.

## Key Points

- Formally measured by the degradation in performance metrics (e.g., BLEU for generation tasks, accuracy for classification) under adversarial perturbations; a robust model shows small degradation.
- Ilyas et al. attribute adversarial vulnerability to non-robust features — high-predictive-power features that are brittle to small input changes — meaning robustness requires learning features that are stable under perturbation.
- Two main defense families: (1) detection-based (identify adversarial inputs and reject/correct them) and (2) model enhancement (adversarial training, certified training, masked training) that modify the model to be intrinsically harder to fool.
- Adversarial training (Madry et al.'s PGD min-max formulation) is the strongest known empirical defense for image models but is computationally expensive; lightweight variants (masked training, FreeLB) trade some robustness for efficiency.
- In the code domain, robustness evaluation requires compilability and functionality-preservation constraints — pure metric degradation is necessary but not sufficient to confirm a valid adversarial example.
- Transferability of adversarial examples (reduced BLEU when examples crafted for model A are applied to model B) is a measure of both attack generality and model fragility to shared non-robust features.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[zhou-2022-adversarial]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[zhou-2022-adversarial]].
