---
type: concept
title: Masked Autoencoding
slug: masked-autoencoding
date: 2026-04-20
updated: 2026-04-20
aliases: [掩码自编码, masked autoencoder]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Masked Autoencoding** (掩码自编码) — a self-supervised training strategy that masks part of the input and learns representations by reconstructing the missing content.

## Key Points

- TimesURL imports the masked autoencoding idea into time-series representation learning as a companion objective to contrastive learning.
- The method uses random masking and reconstructs original values from latent representations produced by the shared encoder.
- Loss is measured only on masked positions, mirroring BERT- and MAE-style selective reconstruction.
- In this paper, masked autoencoding is not the sole objective; it is combined with dual contrastive loss to balance instance-level and segment-level information.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[liu-2023-timesurl]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[liu-2023-timesurl]].
