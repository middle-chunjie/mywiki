---
type: concept
title: Harmonic-Mean Evaluation
slug: harmonic-mean-evaluation
date: 2026-04-20
updated: 2026-04-20
aliases: [adjusted harmonic mean, harmonic mean metric, 调和平均评测]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Harmonic-Mean Evaluation** (调和平均评测) — aggregating task scores with a harmonic mean so systems with severe weaknesses on some tasks are penalized more strongly than under micro or macro averaging.

## Key Points

- The paper uses adjusted harmonic mean as its primary aggregate metric for BBEH because robust general reasoners should not fail badly on a subset of tasks.
- To avoid zero-valued collapse, the authors smooth task accuracies as `a'_i = a_i + 1` before aggregation.
- This choice materially changes rankings: the best general-purpose model reaches only `9.8%` harmonic mean even though its micro average is `23.9%`.
- The metric is explicitly motivated as a better fit than micro or macro average for cross-skill reasoning benchmarks.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[kazemi-2025-bigbench-2502-19187]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[kazemi-2025-bigbench-2502-19187]].
