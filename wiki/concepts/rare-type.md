---
type: concept
title: Rare Type
slug: rare-type
date: 2026-04-20
updated: 2026-04-20
aliases: [long-tail type, 稀有类型]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Rare Type** (稀有类型) — a type whose occurrence frequency in a dataset is so low that purely data-driven predictors have difficulty learning it reliably.

## Key Points

- Following prior work, the paper treats types with proportion below `0.1%` in the dataset as rare types.
- Rare types are a major weakness of neural baselines because they are underrepresented in the training distribution.
- The evaluated datasets show that rare types constitute a large long tail and are dominated by user-defined types.
- HiTYPER improves rare-type prediction by using static constraints that do not depend on frequency counts.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[peng-2022-static-2105-03595]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[peng-2022-static-2105-03595]].
