---
type: concept
title: Invariant Risk Minimization
slug: invariant-risk-minimization
date: 2026-04-20
updated: 2026-04-20
aliases: [IRM]
tags: [domain-generalization, out-of-distribution, invariant-features]
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Invariant Risk Minimization** (不变风险最小化) — a learning framework that seeks a data representation such that the optimal classifier on top of that representation is simultaneously optimal across all training environments, aiming to capture causally stable features rather than spurious correlations.

## Key Points

- Proposed by Arjovsky et al. (2019); formalizes the goal of learning features whose predictive relationship with the label is invariant across environments, rather than minimizing average risk (ERM).
- The training objective adds a penalty term measuring the gradient magnitude of the per-environment loss with respect to a fixed classifier, encouraging the representation to support the same classifier everywhere.
- IRM is motivated by the causal view: invariant features correspond to direct causes of the label and should generalize to new environments, whereas spurious correlates may not hold outside training distributions.
- Empirically, IRM can underperform ERM on many standard benchmarks (e.g., PACS, VLCS) due to optimization difficulties and the need for environment annotations; DAT-based methods like DANN+ELS may surpass IRM on these tasks.

## My Position

<!-- User's stance on this concept. Fed by personal writing; tag "(personal stance)" on such bullets. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[zhang-2023-free-2302-00194]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[zhang-2023-free-2302-00194]].
