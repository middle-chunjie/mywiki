---
type: concept
title: Simultaneous Perturbation Stochastic Approximation
slug: simultaneous-perturbation-stochastic-approximation
date: 2026-04-20
updated: 2026-04-20
aliases: [SPSA, simultaneous perturbation approximation, 同时扰动随机逼近]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Simultaneous Perturbation Stochastic Approximation** (同时扰动随机逼近) — a gradient-estimation method that infers an update direction from loss differences under random perturbations applied to all coordinates at once.

## Key Points

- The paper uses SPSA as the core estimator behind MeZO, with perturbations `z ~ \mathcal{N}(0, I_d)` and projected gradient coefficient `(\ell_+ - \ell_-) / (2\epsilon)`.
- Unlike coordinate-wise finite differences, SPSA needs only two forward evaluations per sample direction, making it viable for large neural networks.
- The authors use `n = 1` by default, arguing that this is the most efficient choice in their preliminary experiments when total forward-pass budget is fixed.
- MeZO's implementation replays the same perturbation from a saved random seed, avoiding the need to store the full perturbation vector in memory.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[malladi-2024-finetuning-2305-17333]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[malladi-2024-finetuning-2305-17333]].
