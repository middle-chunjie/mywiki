---
type: concept
title: Physics-Informed Neural Network
slug: physics-informed-neural-network
date: 2026-04-20
updated: 2026-04-20
aliases: [PINN, 物理约束神经网络]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Physics-Informed Neural Network** (物理约束神经网络) — a neural model trained by minimizing physical residuals and boundary-condition violations in addition to or instead of direct supervised labels.

## Key Points

- The paper evaluates KANs in a PINN setup for a Poisson equation on `[-1,1]^2` with zero Dirichlet boundary conditions.
- Its loss is `loss_pde = \alpha loss_i + loss_b`, using `n_i = 10000` interior points, `n_b = 800` boundary points, and `\alpha = 0.01`.
- On this benchmark, KANs achieve lower `L^2` and `H^1` errors than comparable MLPs with fewer parameters.
- The authors suggest KANs may be useful for PDE model reduction, while also noting that current implementations are slower to train.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[liu-2024-kan-2404-19756]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[liu-2024-kan-2404-19756]].
