---
type: concept
title: Radial Basis Function
slug: radial-basis-function
date: 2026-04-20
updated: 2026-04-20
aliases: [径向基函数, RBF]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Radial Basis Function** (径向基函数) — a nonlinear feature mapping that expands a scalar input into responses centered at multiple anchor points, often to encode distances more smoothly.

## Key Points

- The paper uses Gaussian radial basis functions to expand each inter-atomic distance before graph interaction updates.
- Distances are mapped as `RBF(x) = concat_i exp(-β ||x - μ_i||^2)` over `K` centers spanning the shortest to longest edges in the dataset.
- This expansion converts the raw distance matrix into a tensor `D ∈ R^{N × N × K}` for downstream message passing.
- The authors argue the RBF encoding is more robust and interpretable than feeding distances directly into a simple multilayer perceptron.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[lu-2019-molecular]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[lu-2019-molecular]].
