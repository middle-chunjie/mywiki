---
type: concept
title: Doubly Stochastic Matrix
slug: doubly-stochastic-matrix
date: 2026-04-20
updated: 2026-04-20
aliases: [bistochastic matrix, 双随机矩阵]
tags: []
source_count: 1
confidence: low
domain_volatility: low
last_reviewed: 2026-04-20
---

## Definition

**Doubly Stochastic Matrix** (双随机矩阵) — a non-negative square matrix whose rows and columns each sum to `1`, so it acts as a convex-mixing operator that preserves global mass.

## Key Points

- The paper constrains every residual mixing matrix `H_l^res` in mHC to be doubly stochastic.
- This gives `H_l^res 1_n = 1_n` and `1_n^T H_l^res = 1_n^T`, which the paper uses to argue that average signal intensity is conserved across residual streams.
- The authors emphasize three useful properties: non-expansive spectral norm `<= 1`, closure under multiplication, and an interpretation as convex combinations of stream permutations.
- In mHC, this constraint is what distinguishes stable multi-stream residual propagation from the unconstrained HC design.
- The exact condition degenerates to the scalar `1` when `n = 1`, recovering the ordinary identity mapping of a standard residual connection.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[xie-2026-mhc-2512-24880]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[xie-2026-mhc-2512-24880]].
