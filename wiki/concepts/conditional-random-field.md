---
type: concept
title: Conditional Random Field
slug: conditional-random-field
date: 2026-04-20
updated: 2026-04-20
aliases: [CRF, 条件随机场]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Conditional Random Field** (条件随机场) — a discriminative structured prediction model that scores entire label sequences conditioned on an observed input sequence.

## Key Points

- In [[fang-2021-guided]], the CRF layer is the final decoder on top of Bi-LSTM and guided attention features.
- The CRF consumes concatenated token features `[h_i; h_i']`, where `h_i'` is the interpolation of topic-aware and position-aware representations.
- Its role is to model dependencies among adjacent output labels instead of predicting each token tag independently.
- The paper treats the learning objective as standard sequence tagging, so the CRF improves consistency for multi-token concept boundaries.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[fang-2021-guided]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[fang-2021-guided]].
