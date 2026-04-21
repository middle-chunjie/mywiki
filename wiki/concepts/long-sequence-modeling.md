---
type: concept
title: Long-Sequence Modeling
slug: long-sequence-modeling
date: 2026-04-20
updated: 2026-04-20
aliases: [long sequence modeling, 长序列建模]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Long-Sequence Modeling** (长序列建模) — the design of models and benchmarks where sequence length is large enough that computational scaling and memory compression become central constraints.

## Key Points

- The paper defines a simple regime test using Transformer cost: `FLOPs = 24 D^2 L + 4 D L^2`, with the sequence-dominant threshold `L > 6D`.
- Under this criterion, ImageNet classification with `196` tokens is not a long-sequence task, but dense vision tasks with roughly `4K` tokens are.
- The authors use this distinction to argue that Mamba's linear-time recurrent mixing should matter more for COCO detection and ADE20K segmentation than for ImageNet classification.
- MambaOut's weaker dense-prediction results are presented as indirect evidence that long-sequence visual tasks can still benefit from SSM-style modeling.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[yu-2024-mambaout-2405-07992]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[yu-2024-mambaout-2405-07992]].
