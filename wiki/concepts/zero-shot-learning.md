---
type: concept
title: Zero-Shot Learning
slug: zero-shot-learning
date: 2026-04-20
updated: 2026-04-20
aliases: [零样本学习]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Zero-Shot Learning** (零样本学习) — inference on unseen classes by matching inputs to semantic class descriptions rather than training a supervised classifier for each target label.

## Key Points

- MixCon3D performs zero-shot 3D recognition by encoding class names with the text branch and comparing them against learned 3D or fused image-3D representations.
- The paper uses prompt-engineered class labels to construct the text feature bank `` `F_C^T` `` for downstream benchmarks.
- The model supports point-cloud-only, image-only, and fused inference because the direct point-to-text path is preserved during training.
- Reported zero-shot gains are strongest on Objaverse-LVIS, where MixCon3D reaches `` `52.5%` `` Top1 with PointBERT on the full Ensemble pre-training set.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[gao-2024-sculpting-2311-01734]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[gao-2024-sculpting-2311-01734]].
