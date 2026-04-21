---
type: concept
title: Momentum Encoder
slug: momentum-encoder
date: 2026-04-20
updated: 2026-04-20
aliases: [动量编码器]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Momentum Encoder** (动量编码器) — an encoder whose parameters are updated as an exponential moving average of an online encoder, providing more stable target representations or soft supervision during training.

## Key Points

- [[liu-2025-gear-2501-02772]] adopts a momentum bi-encoder alongside the online GeAR encoders to provide richer soft labels for contrastive learning.
- The paper explicitly cites MoCo and BLIP as the design inspiration for this supervision strategy.
- In GeAR, the momentum update coefficient is `0.995` and the feature queue size is `57600`.
- The soft-label contribution is linearly ramped from `0` to `0.4` over the first `2` epochs.
- The momentum pathway is used during training to improve global query-document alignment rather than as an extra inference-time module.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[liu-2025-gear-2501-02772]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[liu-2025-gear-2501-02772]].
