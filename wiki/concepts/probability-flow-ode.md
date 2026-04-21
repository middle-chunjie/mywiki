---
type: concept
title: Probability Flow ODE
slug: probability-flow-ode
date: 2026-04-20
updated: 2026-04-20
aliases: [概率流常微分方程, probability flow ordinary differential equation]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Probability Flow ODE** (概率流常微分方程) — the deterministic ordinary differential equation whose marginals match those of a diffusion SDE, enabling diffusion sampling without stochastic noise.

## Key Points

- The survey derives probability flow ODE as `dx = {f(x,t) - 0.5 g(t)^2 \nabla_x \log p_t(x)} dt`.
- It emphasizes that this ODE shares marginal densities with the corresponding SDE while allowing larger step sizes.
- The deterministic formulation motivates fast samplers such as DDIM, PNDM, DEIS, and DPM-Solver in the paper's taxonomy.
- The quality-speed trade-off in the survey's benchmark tables is partly organized around how well ODE-based samplers preserve sample quality at lower NFE.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[cao-2023-survey-2209-02646]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[cao-2023-survey-2209-02646]].
