---
type: concept
title: Alternating Optimization
slug: alternating-optimization
date: 2026-04-20
updated: 2026-04-20
aliases: [AO, 交替优化]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Alternating Optimization** (交替优化) — an optimization strategy that solves a coupled problem by repeatedly updating one variable block while holding the others fixed.

## Key Points

- The paper alternates between optimizing the site-selection variable `z` and the perturbation variables `u`.
- Each substep is solved with PGD under its own constraints, reusing the decomposed projection machinery.
- AO costs about twice as much as JO per outer iteration, but the paper reports better empirical convergence and better local optima.
- AO is the strongest unsmoothed solver in the experiments, and AO+RS is the overall best attack generator.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[srikant-2021-generating-2103-11882]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[srikant-2021-generating-2103-11882]].
