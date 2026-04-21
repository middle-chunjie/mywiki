---
type: concept
title: Depthwise Convolution
slug: depthwise-convolution
date: 2026-04-20
updated: 2026-04-20
aliases: [depth-wise convolution, 逐通道卷积]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Depthwise Convolution** (逐通道卷积) — a convolution that applies one spatial filter per channel, reducing cost relative to dense channel-mixing convolutions while preserving local spatial interaction.

## Key Points

- MambaOut uses `7 x 7` depthwise convolution as the token mixer in its Gated CNN block, following ConvNeXt-style design choices.
- The paper applies the depthwise convolution to only part of the hidden channels for better practical throughput, following partial-convolution ideas from InceptionNeXt-like designs.
- This choice is important because it gives MambaOut a strong local inductive bias without relying on SSM for token mixing.
- The paper's empirical point is that such a simple convolutional mixer already outperforms several visual Mamba alternatives on ImageNet.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[yu-2024-mambaout-2405-07992]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[yu-2024-mambaout-2405-07992]].
