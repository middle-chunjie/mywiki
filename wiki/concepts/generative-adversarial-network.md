---
type: concept
title: Generative Adversarial Network
slug: generative-adversarial-network
date: 2026-04-20
updated: 2026-04-20
aliases: [GAN, 生成对抗网络]
tags: [generative-model, adversarial-training, deep-learning]
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Generative Adversarial Network** (生成对抗网络) — a framework in which a generator and a discriminator are jointly trained via a minimax game: the generator learns to produce samples indistinguishable from real data, while the discriminator learns to tell real from generated samples.

## Key Points

- Originally proposed by Goodfellow et al. (2014) for image generation; the minimax objective is `min_G max_D E[log D(x)] + E[log(1 - D(G(z)))]`.
- The adversarial dynamic produces increasingly realistic samples because the generator's loss is directly shaped by the discriminator's evolving discrimination ability.
- Applied to IR in IRGAN (Wang et al., 2017), which replaces image generation with document retrieval: a generative retrieval model produces documents that a discriminative model must distinguish from ground-truth relevant documents.
- AR2 adapts the GAN minimax framework for dense text retrieval by replacing the generative model with a dual-encoder retriever and the discriminator with a cross-encoder ranker, enabling the framework to support fast ANN-based document indexing.
- A key limitation of the original IRGAN for dense retrieval is the absence of a dual-encoder architecture needed for efficient document indexing; AR2 addresses this directly.

## My Position

<!-- User's stance on this concept. Fed by personal writing; tag "(personal stance)" on such bullets. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[zhang-2022-adversarial]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[zhang-2022-adversarial]].
