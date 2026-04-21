---
type: concept
title: Projected Gradient Descent
slug: projected-gradient-descent
date: 2026-04-20
updated: 2026-04-20
aliases: [PGD, 投影梯度下降]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Projected Gradient Descent** (投影梯度下降) — an iterative constrained optimization method that takes a gradient step on the objective and then projects the updated variables back into the feasible set.

## Key Points

- The paper uses PGD as the solver for the relaxed adversarial program optimization problem.
- Binary site and perturbation variables are relaxed to continuous boxes before optimization, then projected back to satisfy budget and simplex-like constraints.
- The projection decomposes into separate subproblems over `z` and each `u_i`, which are solved with bisection-based root finding.
- PGD appears both in the joint optimizer and as the inner update rule for the alternating optimizer.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[srikant-2021-generating-2103-11882]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[srikant-2021-generating-2103-11882]].
