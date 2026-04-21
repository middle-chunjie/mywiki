---
type: concept
title: Tree-Structured Parzen Estimator
slug: tree-structured-parzen-estimator
date: 2026-04-20
updated: 2026-04-20
aliases: [TPE, tree structured parzen estimator, 树结构 Parzen 估计器]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Tree-Structured Parzen Estimator** (树结构 Parzen 估计器) — a Bayesian optimization method that models densities over good and bad configurations and ranks candidates by their relative likelihood under those densities.

## Key Points

- LLAMBO uses TPE as both a baseline and a conceptual starting point for its generative surrogate and candidate sampler.
- The paper rewrites the TPE density-ratio objective through Bayes' rule into an LLM-friendly classification score `p(s <= tau | h)`.
- Candidate sampling in LLAMBO is inspired by TPE but conditions on a desired target value rather than only a quantile split between good and bad points.
- Experiments compare against both independent and multivariate TPE variants.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[unknown-nd-large-2402-03921]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[unknown-nd-large-2402-03921]].
