---
type: concept
title: B-Spline
slug: b-spline
date: 2026-04-20
updated: 2026-04-20
aliases: [B-spline, B样条]
tags: []
source_count: 1
confidence: low
domain_volatility: low
last_reviewed: 2026-04-20
---

## Definition

**B-Spline** (B样条) — a locally supported piecewise-polynomial basis used to represent smooth one-dimensional functions through learnable coefficients on a grid.

## Key Points

- The paper parameterizes each KAN edge function's spline branch as `spline(x) = \sum_i c_i B_i(x)`, where the coefficients `c_i` are trained.
- KANs usually use cubic splines with order `k = 3`, trading smoothness against optimization stability.
- Because B-splines have local support, the paper can update grids adaptively during training and motivate a locality-based story for avoiding [[catastrophic-forgetting]].
- Grid extension transfers a coarse spline to a finer grid by least-squares matching, making spline resolution an explicit scaling knob.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[liu-2024-kan-2404-19756]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[liu-2024-kan-2404-19756]].
