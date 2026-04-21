---
type: concept
title: Bayesian Optimization
slug: bayesian-optimization
date: 2026-04-20
updated: 2026-04-20
aliases: [BO, bayesian optimization, 贝叶斯优化]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Bayesian Optimization** (贝叶斯优化) — a sample-efficient strategy for optimizing expensive black-box functions by fitting a surrogate model and using an acquisition rule to choose future evaluations.

## Key Points

- The paper formalizes BO as finding `h* = arg min_h f(h)` when direct gradients are unavailable and evaluations are costly.
- LLAMBO augments three BO stages with an LLM: warmstarting, surrogate modeling, and candidate sampling.
- The method keeps the classic BO loop of model update, acquisition scoring, and iterative evaluation, but changes how priors and candidate proposals are produced.
- The paper argues that BO is particularly vulnerable when observations are sparse, making it a natural testbed for LLM priors and in-context learning.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[unknown-nd-large-2402-03921]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[unknown-nd-large-2402-03921]].
