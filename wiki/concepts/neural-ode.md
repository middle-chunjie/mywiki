---
type: concept
title: Neural ODE
slug: neural-ode
date: 2026-04-20
updated: 2026-04-20
aliases: [neural ordinary differential equation, 神经常微分方程]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Neural ODE** (神经常微分方程) — a neural network formulation in which hidden states evolve according to a parameterized ordinary differential equation and are obtained by numerical integration.

## Key Points

- The paper uses a Neural ODE to generate positional representations continuously instead of learning an independent embedding vector for each token position.
- FLOATER parameterizes the dynamics as `dp(t)/dt = h(t, p(t); θ_h)` and samples discrete position vectors at `t_i = i·Δ`.
- This formulation makes sinusoidal encoding a special case of the learned dynamical system, so the method is compatible with vanilla Transformer positional design.
- The appendix uses different ODE solvers by task, including Runge-Kutta on WMT and a midpoint method on longer-context benchmarks.
- The authors rely on Neural ODE gradient computation to train the positional dynamics jointly with the Transformer.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[liu-2020-encode-2003-09229]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[liu-2020-encode-2003-09229]].
