---
type: concept
title: Overfitting
slug: overfitting
date: 2026-04-20
updated: 2026-04-20
aliases: [过拟合]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Overfitting** (过拟合) — the phenomenon where a model fits training data too aggressively and loses generalization performance on held-out evaluation data.

## Key Points

- The paper presents overfitting as a core motivation for avoiding full fine-tuning of large CLIP backbones on text-video retrieval datasets.
- On MSR-VTT, the fully fine-tuned baseline shows rapidly decreasing training loss but validation `R@1` peaks early and then degrades.
- The cross-modal adapter trains far fewer parameters and exhibits a healthier alignment between lower training loss and better validation retrieval.
- The authors interpret the result as evidence that small retrieval datasets are not ideal for updating more than `123M` parameters end to end.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[jiang-2022-crossmodal-2211-09623]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[jiang-2022-crossmodal-2211-09623]].
