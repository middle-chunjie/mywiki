---
type: concept
title: Warmstarting
slug: warmstarting
date: 2026-04-20
updated: 2026-04-20
aliases: [warm start, 热启动]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Warmstarting** (热启动) — initializing an optimization procedure with informative starting points before the standard iterative search dynamics begin.

## Key Points

- LLAMBO uses zero-shot prompting to propose the first `5` configurations before normal BO iterations start.
- The paper studies no-context, partial-context, and full-context warmstarts, showing stronger hyperparameter correlations as more dataset information is added.
- Unlike meta-learning-based warmstarts, the method does not require a database of prior solved tasks.
- The reported benefit is strongest in the earliest trials, where a better initial pool materially improves regret.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[unknown-nd-large-2402-03921]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[unknown-nd-large-2402-03921]].
