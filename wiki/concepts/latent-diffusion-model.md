---
type: concept
title: Latent Diffusion Model
slug: latent-diffusion-model
date: 2026-04-20
updated: 2026-04-20
aliases: [潜空间扩散模型, latent diffusion]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Latent Diffusion Model** (潜空间扩散模型) — a diffusion model that performs denoising in a learned latent space rather than directly in the original observation space.

## Key Points

- The survey describes latent diffusion as a way to make diffusion learning and sampling easier than operating directly in pixel space.
- It highlights joint objectives that combine encoder, decoder, and denoising score-matching losses over latent variables `z_t`.
- LSGM and Stable Diffusion are presented as representative examples of latent-space diffusion design.
- The paper positions latent diffusion as one branch of the broader "diffusion process design" category.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[cao-2023-survey-2209-02646]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[cao-2023-survey-2209-02646]].
