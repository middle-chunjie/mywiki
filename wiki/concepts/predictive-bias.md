---
type: concept
title: Predictive Bias
slug: predictive-bias
date: 2026-04-20
updated: 2026-04-20
aliases: [prediction bias, 预测偏差]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Predictive Bias** (预测偏差) — the skew in a model's output distribution toward particular labels or attributes that arises from the prompt or context rather than the actual input content.

## Key Points

- The paper treats predictive bias as a property of the prompt itself rather than only of the model or dataset.
- It estimates this bias by querying the model with a content-free input and examining the induced class distribution.
- Lower predictive bias corresponds to a more uniform label distribution and is associated with better downstream ICL performance.
- The proposed prompt-search algorithms use predictive bias as the main objective for selecting both demonstrations and their order.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[ma-2023-fairnessguided-2303-13217]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[ma-2023-fairnessguided-2303-13217]].
