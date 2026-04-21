---
type: concept
title: Expectation-Maximization
slug: expectation-maximization
date: 2026-04-20
updated: 2026-04-20
aliases: [EM, expectation maximization, 期望最大化]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Expectation-Maximization** (期望最大化) — an iterative optimization procedure for latent-variable models that alternates between estimating hidden assignments and updating parameters.

## Key Points

- The paper uses EM to partition generated QA pairs into four credit-score categories: low, relatively low, medium, and high.
- The low and high extremes are filtered out because they are respectively too easy or too noisy.
- The remaining middle categories provide the candidate pool for later graph-based conversation assembly.
- EM is used here as a data selection mechanism rather than as the core learning objective of the final CQA model.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[li-2023-sm-2312-16511]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[li-2023-sm-2312-16511]].
