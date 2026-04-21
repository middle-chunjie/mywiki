---
type: concept
title: Diffusion Model
slug: diffusion-model
date: 2026-04-20
updated: 2026-04-20
aliases: [扩散模型, diffusion probabilistic model]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Diffusion Model** (扩散模型) — a generative model that learns to reverse a gradual corruption process, typically transforming a simple prior distribution back into data through iterative denoising.

## Key Points

- The paper frames diffusion models around a forward corruption process from `x_0` to a simple prior such as Gaussian noise and a learned reverse process that reconstructs data.
- It treats both discrete DDPM formulations and continuous-time score-SDE formulations as part of the same broader diffusion-model family.
- The survey emphasizes diffusion models' strong sample quality relative to VAEs, EBMs, GANs, and normalizing flows, while also highlighting their slow iterative sampling cost.
- A central theme of the survey is that later work modifies either the sampler, the forward process, the training objective, or the source-target endpoints of the diffusion trajectory.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[cao-2023-survey-2209-02646]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[cao-2023-survey-2209-02646]].
