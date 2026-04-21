---
type: entity
title: MATH
slug: math-benchmark
date: 2026-04-20
entity_type: benchmark
aliases: [MATH, Hendrycks MATH, MATH benchmark]
tags: []
---

## Description

MATH is the mathematical reasoning benchmark used in [[snell-2024-scaling-2408-03314]] to evaluate verifier search, revision policies, and compute-optimal inference allocation.

## Key Contributions

- Supplies the `12k` train / `500` test split used in the paper's main experiments.
- Exposes difficulty-dependent differences between search-heavy and revision-heavy test-time strategies.
- Serves as the basis for the paper's FLOPs-matched comparison between adaptive inference and parameter scaling.

## Related Concepts

- [[question-difficulty]]
- [[best-of-n-sampling]]
- [[compute-optimal-test-time-scaling]]

## Sources

- [[snell-2024-scaling-2408-03314]]
