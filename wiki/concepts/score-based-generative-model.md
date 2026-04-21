---
type: concept
title: Score-Based Generative Model
slug: score-based-generative-model
date: 2026-04-20
updated: 2026-04-20
aliases: [基于评分的生成模型, score model]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Score-Based Generative Model** (基于评分的生成模型) — a generative model that learns the score function `\nabla_x \log p_t(x)` of perturbed data distributions and uses it to simulate reverse stochastic or deterministic dynamics.

## Key Points

- The survey places score-SDE models and DDPMs in a unified landscape by relating score estimation to noise-prediction parameterizations.
- It describes the reverse-time SDE as depending on the score term `\nabla_x \log p_t(x)`, estimated by a neural network `s_{\theta}(x,t)`.
- The denoising score-matching objective is presented as the central training principle for continuous-time diffusion models.
- Score-based formulations are important in the survey because they directly motivate probability-flow ODE samplers and extensions to manifolds, graphs, and function spaces.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[cao-2023-survey-2209-02646]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[cao-2023-survey-2209-02646]].
