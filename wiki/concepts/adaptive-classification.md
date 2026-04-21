---
type: concept
title: Adaptive Classification
slug: adaptive-classification
date: 2026-04-20
updated: 2026-04-20
aliases: [adaptive classification, 自适应分类]
tags: [classification, efficiency]
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Adaptive Classification** (自适应分类) — a classification strategy that allocates variable computation or representation capacity per instance, usually escalating only hard examples to higher-capacity predictors.

## Key Points

- In this paper, adaptive classification uses nested prefixes of one MRL embedding rather than multiple separately encoded models.
- The cascade advances through dimensions such as `8 -> 16 -> 32` based on thresholds over the maximum softmax probability on a validation set.
- On ImageNet-1K, the approach matches `76.30%` accuracy of a fixed `512`-dimensional baseline with only about `37` expected dimensions.
- The result suggests that class and instance difficulty vary substantially, so a fixed embedding width is often wasteful.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[kusupati-2024-matryoshka-2205-13147]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[kusupati-2024-matryoshka-2205-13147]].
