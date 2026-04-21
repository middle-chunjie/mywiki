---
type: concept
title: Bilevel Optimization
slug: bilevel-optimization
date: 2026-04-20
updated: 2026-04-20
aliases: [bi-level optimization, 双层优化]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Bilevel Optimization** (双层优化) — an optimization setup with nested objectives where an outer problem optimizes parameters that indirectly shape the solution of an inner learning problem.

## Key Points

- The paper formulates MTTT as a bilevel problem in which the inner loop adapts encoder weights `W_t`, while the outer loop learns `theta = (theta_g, theta_h, theta_phi, theta_psi, W_0)`.
- The outer objective is the supervised loss after adaptation, `L_T(theta; X, y) = L(h(f(psi(X); W_T)), y)`, so gradients must pass through the inner-loop updates.
- This formulation turns the design of a self-supervised test-time objective from heuristic task engineering into direct optimization for downstream prediction quality.
- The implementation uses automatic differentiation through gradients of gradients, which the paper reports is practical in JAX but still infrastructure-limited.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[sun-2024-learn-2310-13807]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[sun-2024-learn-2310-13807]].
