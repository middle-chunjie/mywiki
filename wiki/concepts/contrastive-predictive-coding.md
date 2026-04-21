---
type: concept
title: Contrastive Predictive Coding
slug: contrastive-predictive-coding
date: 2026-04-20
updated: 2026-04-20
aliases: [CPC, 对比预测编码]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Contrastive Predictive Coding** (对比预测编码) — a self-supervised learning framework that learns representations by predicting future latent structure with a contrastive density-ratio objective instead of reconstructing raw observations.

## Key Points

- CPC first encodes observations into latent vectors `z_t` and then summarizes past latents into a context state `c_t` with an autoregressive model.
- The prediction target is future latent information rather than raw future input, which reduces the burden of full generative modeling.
- The scoring function uses a bilinear form `` `exp(z_{t+k}^T W_k c_t)` `` and is trained with an InfoNCE objective over one positive and multiple negatives.
- The paper applies the same high-level recipe to speech, images, text, and reinforcement learning, changing only the modality-specific encoder and autoregressive backbone.
- Empirically, CPC yields strong linear-probe or transfer performance across all four domains, especially on LibriSpeech and ImageNet.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[oord-2019-representation-1807-03748]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[oord-2019-representation-1807-03748]].
