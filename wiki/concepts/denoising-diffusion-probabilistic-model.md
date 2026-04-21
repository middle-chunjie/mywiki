---
type: concept
title: Denoising Diffusion Probabilistic Model
slug: denoising-diffusion-probabilistic-model
date: 2026-04-20
updated: 2026-04-20
aliases: [去噪扩散概率模型, DDPM]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Denoising Diffusion Probabilistic Model** (去噪扩散概率模型) — a diffusion model that defines a Markov forward noising process and learns Gaussian reverse transitions to denoise samples step by step.

## Key Points

- The survey uses DDPM as the canonical discrete-time formulation for diffusion models.
- Its forward process is written as `q(x_{1:T}|x_0) = \prod_{t=1}^{T} q(x_t|x_{t-1})`, while the reverse process is parameterized by Gaussian transitions `p_{\theta}(x_{t-1}|x_t)`.
- Training is described through a variational bound that decomposes into prior, denoising, and reconstruction terms before being simplified to a noise-prediction objective.
- The paper treats many later samplers and accelerators as refinements of the DDPM reverse process rather than departures from the DDPM learning setup.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[cao-2023-survey-2209-02646]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[cao-2023-survey-2209-02646]].
