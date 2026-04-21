---
type: concept
title: Uncertainty Sampling
slug: uncertainty-sampling
date: 2026-04-20
updated: 2026-04-20
aliases: [uncertainty-based sampling, 不确定性采样]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Uncertainty Sampling** (不确定性采样) — an active learning strategy that prioritizes examples on which the current model is least certain.

## Key Points

- The paper evaluates entropy-based heuristics, BALD, variation ratios, and Bayesian uncertainty baselines for CNN active learning.
- Its central empirical claim is that these methods degrade in the batch setting because selected samples become highly correlated.
- Random sampling can outperform uncertainty-based methods once batch correlation is taken into account.
- The authors position geometric coverage as a more suitable acquisition principle for CNNs than raw softmax uncertainty.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[sener-2018-active-1708-00489]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[sener-2018-active-1708-00489]].
