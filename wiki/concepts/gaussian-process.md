---
type: concept
title: Gaussian Process
slug: gaussian-process
date: 2026-04-20
updated: 2026-04-20
aliases: [GP, gaussian process, 高斯过程]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Gaussian Process** (高斯过程) — a probabilistic nonparametric model over functions that is widely used as a surrogate model in Bayesian optimization because it provides both predictions and uncertainty estimates.

## Key Points

- The paper treats GP as a strong BO baseline for both warmstarting experiments and surrogate-model comparisons.
- GP achieves better-calibrated uncertainty than the discriminative LLAMBO surrogate, especially on log predictive density and empirical coverage.
- LLAMBO nevertheless reports better prediction and regret behavior in low-observation settings, showing a trade-off between calibration and exploitation quality.
- The work positions GP as the standard principled probabilistic reference point for evaluating LLM-based surrogates.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[unknown-nd-large-2402-03921]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[unknown-nd-large-2402-03921]].
