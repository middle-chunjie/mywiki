---
type: concept
title: Vector Scaling
slug: vector-scaling
date: 2026-04-20
updated: 2026-04-20
aliases: [diagonal scaling, 向量缩放]
tags: [calibration, probability-estimation]
source_count: 1
confidence: low
domain_volatility: low
last_reviewed: 2026-04-20
---

## Definition

**Vector Scaling** (向量缩放) — a restricted form of affine probability calibration where the weight matrix `W` in `q = softmax(W * p + b)` is constrained to be diagonal, so each class probability is independently rescaled rather than using a full matrix transformation.

## Key Points

- A special case of the Platt scaling / temperature scaling family: Platt uses a full 2×2 matrix for binary tasks; temperature scaling uses a single scalar; vector scaling applies a per-class scalar, growing linearly rather than quadratically in the number of classes.
- The diagonal constraint is critical for tractability in generation settings where `p` spans the full vocabulary (~50,000 tokens); a full matrix would require ~2.5 billion parameters for GPT-3's vocabulary.
- In [[contextual-calibration]], the diagonal entries of `W` are set to `diag(p_cf)^{-1}` (the reciprocal of content-free-input probabilities), requiring zero labeled data.
- Originally introduced in the neural network calibration literature (Guo et al., ICML 2017) for post-hoc calibration of modern classifiers with held-out validation data.

## My Position

<!-- User's stance on this concept. Fed by personal writing; tag "(personal stance)" on such bullets. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[zhao-2021-calibrate-2102-09690]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[zhao-2021-calibrate-2102-09690]].
