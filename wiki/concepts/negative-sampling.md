---
type: concept
title: Negative Sampling
slug: negative-sampling
date: 2026-04-20
updated: 2026-04-20
aliases: [负采样]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Negative Sampling** (负采样) — a training strategy that contrasts a positive target against sampled non-target examples so the model learns discriminative structure without full normalization.

## Key Points

- CPC draws one positive future sample and `N-1` negatives from the proposal distribution for each contrastive prediction.
- In the audio experiments, negative-sample composition materially changes phone-classification quality, showing the design is not incidental.
- The paper compares mixed-speaker, same-speaker, exclusion, and within-sequence negative pools for speech.
- Negative sampling makes CPC computationally tractable in settings where direct likelihood modeling would be costly.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[oord-2019-representation-1807-03748]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[oord-2019-representation-1807-03748]].
