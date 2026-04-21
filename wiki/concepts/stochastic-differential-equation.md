---
type: concept
title: Stochastic Differential Equation
slug: stochastic-differential-equation
date: 2026-04-20
updated: 2026-04-20
aliases: [随机微分方程, SDE]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Stochastic Differential Equation** (随机微分方程) — a differential equation driven by both deterministic drift and stochastic noise terms, used in diffusion models to describe continuous-time forward and reverse processes.

## Key Points

- The survey writes diffusion dynamics in continuous time as `dx = f(x,t)dt + g(t)dw`.
- It highlights VP and VE SDEs as two standard forward-process families within score-based generative modeling.
- Reverse-time sampling is expressed as another SDE that subtracts `g(t)^2 \nabla_x \log p_t(x)` from the drift.
- This SDE view is the bridge the paper uses to connect DDPMs with more general samplers and non-Euclidean diffusion settings.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[cao-2023-survey-2209-02646]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[cao-2023-survey-2209-02646]].
