---
type: concept
title: Time Reconstruction
slug: time-reconstruction
date: 2026-04-20
updated: 2026-04-20
aliases: [时间重建, temporal reconstruction]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Time Reconstruction** (时间重建) — recovering the original time-series signal from masked or corrupted observations to force an encoder to retain detailed temporal information.

## Key Points

- TimesURL adds reconstruction because contrastive learning alone is argued to undercapture instance-level information needed by some downstream tasks.
- The encoder processes masked series and their augmented counterparts, and the decoder reconstructs the original full signals.
- Reconstruction loss is computed with MSE only on masked timestamps, following the masked autoencoding pattern.
- Ablation shows removing time reconstruction lowers average UEA accuracy from `0.752` to `0.735`.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[liu-2023-timesurl]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[liu-2023-timesurl]].
