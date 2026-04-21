---
type: concept
title: Bias Metric
slug: bias-metric
date: 2026-04-20
updated: 2026-04-20
aliases: [bias measure, 偏见度量]
tags: [evaluation, fairness]
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Bias Metric** (偏见度量) — a quantitative statistic used to measure the prevalence or disparity of biased behavior in model outputs.

## Key Points

- [[liu-nd-uncovering]] introduces three complementary metrics: Code Bias Score, UnFairness Score, and the standard deviation of demographic frequencies.
- CBS measures overall biased-code prevalence, while UFS measures directional disparity between a selected demographic pair.
- The paper uses standard deviation over all valid demographics to capture broader imbalance beyond a single pairwise comparison.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[liu-nd-uncovering]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[liu-nd-uncovering]].
