---
type: concept
title: Gated CNN
slug: gated-cnn
date: 2026-04-20
updated: 2026-04-20
aliases: [gated convolutional network, 门控卷积网络]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Gated CNN** (门控卷积网络) — a convolutional block that combines learned gating with convolutional token mixing, here serving as the simplified backbone primitive from which MambaOut is built.

## Key Points

- The paper treats the Mamba block as an extension of the Gated CNN block with an added SSM token mixer.
- In the shared formulation, the block computes `Y = (TokenMixer(X'W_1) \odot \sigma(X'W_2)) W_3 + X`, with gating and residual addition.
- MambaOut removes the SSM and keeps the Gated CNN-style mixer, turning the block into a controlled ablation of Mamba for visual tasks.
- The resulting Gated CNN stack is sufficient to beat visual Mamba baselines on ImageNet, which is central to the paper's argument.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[yu-2024-mambaout-2405-07992]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[yu-2024-mambaout-2405-07992]].
