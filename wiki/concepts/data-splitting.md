---
type: concept
title: Data Splitting
slug: data-splitting
date: 2026-04-20
updated: 2026-04-20
aliases: [dataset split strategy, partitioning strategy, 数据划分]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Data Splitting** (数据划分) — the strategy used to partition a dataset into training, validation, and test subsets, thereby controlling leakage and generalization difficulty.

## Key Points

- The paper compares method-level, class-level, and project-level splits for code summarization datasets.
- Models consistently perform better on method-level splits than on project-level splits, indicating that random splitting can make evaluation easier.
- In the reported experiments, changing split strategy affects absolute performance but leaves the relative ranking of models mostly intact.
- Split granularity is therefore part of the evaluation protocol and should be reported explicitly when comparing systems.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[shi-2022-evaluation-2107-07112]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[shi-2022-evaluation-2107-07112]].
