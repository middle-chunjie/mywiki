---
type: concept
title: Adversarial Training
slug: adversarial-training
date: 2026-04-20
updated: 2026-04-20
aliases: [Robust Training, 对抗训练]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Adversarial Training** (对抗训练) — a robustness strategy that augments model training with adversarial or perturbed examples so the learned decision boundary becomes harder to exploit.

## Key Points

- RoPGen instantiates adversarial training for code authorship attribution using both style imitation and style perturbation.
- The paper compares RoPGen against Basic-AT-AE, Basic-AT-COM, and PGD-AT baselines.
- Basic adversarial training with only adversarial examples helps against adversarial-example attacks but does not reliably defend coding-style imitation or hiding.
- The paper argues RoPGen outperforms baseline adversarial-training variants because it combines data augmentation with gradient augmentation rather than merely enlarging the training set.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[li-2022-ropgen-2202-06043]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[li-2022-ropgen-2202-06043]].
