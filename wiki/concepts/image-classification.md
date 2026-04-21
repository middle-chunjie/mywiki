---
type: concept
title: Image Classification
slug: image-classification
date: 2026-04-20
updated: 2026-04-20
aliases: [图像分类]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Image Classification** (图像分类) — the task of assigning one or more semantic class labels to an input image.

## Key Points

- The paper argues that standard ImageNet classification is neither long-sequence nor autoregressive, making it a poor fit for Mamba's core strengths.
- With `224 x 224` inputs and `16 x 16` patches, the token length is only `196`, far below the paper's long-sequence threshold.
- MambaOut uses a DeiT-style training recipe with `300` epochs, AdamW, `lr = 4e-3`, batch size `4096`, Mixup, CutMix, and label smoothing.
- MambaOut beats the compared visual Mamba models on ImageNet across model scales, supporting the claim that SSM is unnecessary for this task.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[yu-2024-mambaout-2405-07992]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[yu-2024-mambaout-2405-07992]].
