---
type: concept
title: Expected Improvement
slug: expected-improvement
date: 2026-04-20
updated: 2026-04-20
aliases: [EI, expected improvement, 期望改进]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Expected Improvement** (期望改进) — an acquisition function that scores a candidate by the expected gain it could achieve relative to the current best observed objective value.

## Key Points

- The end-to-end LLAMBO pipeline uses expected improvement to pick the next configuration from the sampled candidate pool.
- In the paper's formulation, `a(h) = E[max(p(s | h) - f(h_best), 0)]` using the surrogate model's predictive distribution.
- EI is the bridge between the LLM-generated candidate set and the actual evaluation decision in the BO loop.
- Regret analyses in the paper examine how well different surrogates and samplers support EI-based selection when observations are sparse.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[unknown-nd-large-2402-03921]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[unknown-nd-large-2402-03921]].
