---
type: concept
title: Variational Inference
slug: variational-inference
date: 2026-04-20
updated: 2026-04-20
aliases: [VI, 变分推断]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Variational Inference** (变分推断) — an approximate Bayesian inference method that fits a tractable surrogate distribution to an intractable posterior by optimizing an evidence lower bound.

## Key Points

- The paper uses mean-field variational inference to estimate latent subject skills and item parameters for DAD leaderboards.
- The variational family factorizes over latent variables, with Gaussian factors for `\theta`, `\beta`, and `\gamma`, and Gamma factors for precision variables.
- Optimization minimizes KL divergence to the true posterior, equivalently maximizing the ELBO with ADAM.
- This inference choice makes the Bayesian IRT formulation practical at the scale of `1.9M` subject-item response pairs.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[rodriguez-2021-evaluation]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[rodriguez-2021-evaluation]].
