---
type: concept
title: Majority Label Bias
slug: majority-label-bias
date: 2026-04-20
updated: 2026-04-20
aliases: [majority class bias, 多数标签偏差]
tags: [bias, few-shot-learning, in-context-learning]
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Majority Label Bias** (多数标签偏差) — the tendency of autoregressive language models used in few-shot in-context learning to over-predict whichever class label appears most frequently in the prompt's training examples, independent of the actual test input.

## Key Points

- When one class is more frequent in the few-shot prompt (e.g., 3 Positive vs. 1 Negative), GPT-3 2.7B skews its predictions heavily toward that class, causing accuracy degradation on balanced evaluation sets.
- The bias also causes accuracy to *drop* when moving from 0-shot to 1-shot (the model simply repeats the single training example's label); for 4-shot LAMA, 50.2% of predictions are a repeat of one of the four training answers vs. a ground-truth repeat rate of 24.7%.
- Interacts with [[recency-bias]]: even a balanced prompt can exhibit majority-label-like effects if one class clusters at the end.
- Mitigated by [[contextual-calibration]], which normalizes the output distribution using a content-free input rather than requiring balanced label selection in the prompt.

## My Position

<!-- User's stance on this concept. Fed by personal writing; tag "(personal stance)" on such bullets. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[zhao-2021-calibrate-2102-09690]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[zhao-2021-calibrate-2102-09690]].
