---
type: concept
title: Unimodality-Supervised Multimodal Contrastive Learning
slug: unimodality-supervised-multimodal-contrastive-learning
date: 2026-04-20
updated: 2026-04-20
aliases: [UniS-MMC, unimodality-supervised MMC, 单模态监督多模态对比学习]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Unimodality-Supervised Multimodal Contrastive Learning** (单模态监督多模态对比学习) — a multimodal contrastive learning framework that uses per-sample unimodal prediction correctness as weak supervision to distinguish positive, semi-positive, and negative cross-modal pairs, thereby aligning ineffective modality representations toward effective ones.

## Key Points

- Unlike unsupervised MMC (same-sample pairs are always positive) and supervised MMC (same-label pairs are positive), UniS-MMC creates three pair types based on unimodal prediction outcomes: positive (both correct), semi-positive (one correct, one wrong), and negative (both wrong).
- For semi-positive pairs the gradient of the correct-modality representation is **detached** so only the ineffective representation is updated, preventing collapse of the effective representation's geometry.
- The negative pair setting pushes representations of jointly-misclassified samples apart, encouraging the model to capture complementary rather than redundant inter-modal information.
- The total loss combines unimodal prediction loss `L_{uni}`, multimodal classification loss `L_{multi}`, and the contrastive term `λ·L_{mmc}` with `λ = 0.1` and temperature `τ = 0.07`.
- The method preserves the bimodal sub-cluster structure in embedding space (two distribution clouds per class), unlike prior contrastive methods that merge both modalities into one.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[zou-2023-unismmc-2305-09299]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[zou-2023-unismmc-2305-09299]].
