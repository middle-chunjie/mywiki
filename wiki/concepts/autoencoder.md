---
type: concept
title: Autoencoder
slug: autoencoder
date: 2026-04-20
updated: 2026-04-20
aliases: [自编码器]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Autoencoder** (自编码器) — a model composed of an encoder and decoder that learns representations by reconstructing its input or a transformed version of it.

## Key Points

- The paper uses a Y-shaped architecture where `f` is the shared encoder, `g` is the reconstruction decoder, and `h` is the main-task head.
- The self-supervised branch reconstructs each token `x_i` from a transformed input `phi(x_i)`, making reconstruction the auxiliary task used for test-time adaptation.
- During TTT, only the shared encoder parameters `W` are updated; the decoder `g` is treated as fixed inside the inner loop even though it is learned in the outer loop.
- For `MTTT-MLP`, the paper adds Decoder LN after `g`, and the ablation shows this normalization materially improves downstream accuracy.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[sun-2024-learn-2310-13807]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[sun-2024-learn-2310-13807]].
