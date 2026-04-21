---
type: concept
title: Gaussian Mixture Model
slug: gaussian-mixture-model
date: 2026-04-20
updated: 2026-04-20
aliases: [GMM, 高斯混合模型]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Gaussian Mixture Model** (高斯混合模型) — a probabilistic model that represents a distribution as a weighted mixture of multiple Gaussian components.

## Key Points

- The paper uses a two-component GMM over reconstruction losses to separate qualified from unqualified comments.
- The modeled density is `P(x) = \pi N(x|\mu_q,\sigma_q) + (1-\pi)N(x|\mu_{uq},\sigma_{uq})`.
- The mixture formalizes the trade-off between high-quality retained queries and noisy discarded comments.
- GMM is preferred over fixed percentile cutoffs because it adapts the split to the empirical loss distribution.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[sun-2022-importance-2202-06649]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[sun-2022-importance-2202-06649]].
