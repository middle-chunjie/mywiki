---
type: concept
title: Variational AutoEncoder
slug: variational-autoencoder
date: 2026-04-20
updated: 2026-04-20
aliases: [VAE, 变分自编码器]
tags: [generative-model, representation-learning, deep-learning]
source_count: 1
confidence: low
domain_volatility: low
last_reviewed: 2026-04-20
---

## Definition

**Variational AutoEncoder** (变分自编码器) — a generative latent-variable model that learns to encode input `x` into a distribution `q(z|x) = N(μ(x), σ(x))` and decode sampled latent vectors `z` back to the input space, trained jointly by minimizing reconstruction loss plus a KL-divergence regularizer on the latent distribution.

## Key Points

- The encoder maps input `x` to parameters `μ` and `σ` of a Gaussian; the reparameterization trick (`z = μ + σ·ε`, `ε ~ N(0,I)`) enables gradient-based training through the sampling step.
- Training objective: `L(f,g) = Σ_i {−D_KL[q(z|x_i)||p(z)] + E_{q(z|x_i)}[ln p(x_i|z)]}`, balancing reconstruction fidelity and latent space regularity.
- In the CWI domain-adaptation setting, the VAE encoder provides latent features `F_v` that augment the BiLSTM and Transformer features; the reconstruction task serves as a regularizing auxiliary objective scaled by `α = 0.1`.
- VAE features offer modest and inconsistent gains: on CompLex LCP single-word test, Base+VAE+DA (`.7554`) slightly underperforms Base+DA (`.7569`), but Base+VAE+LA outperforms on French MAE in CWI 2018 cross-lingual test.
- Used primarily as a feature augmentation mechanism rather than for generation; the latent space is not sampled at inference.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[zaharia-2022-domain-2205-07283]]
- [[sun-2022-importance-2202-06649]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[zaharia-2022-domain-2205-07283]].
