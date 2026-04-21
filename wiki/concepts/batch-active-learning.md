---
type: concept
title: Batch Active Learning
slug: batch-active-learning
date: 2026-04-20
updated: 2026-04-20
aliases: [批量主动学习]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Batch Active Learning** (批量主动学习) — an active learning setting in which the learner acquires labels for a subset of examples at each round rather than querying one example at a time.

## Key Points

- The paper treats batch acquisition as the realistic regime for CNNs because retraining after every single query is computationally intractable.
- It argues that batch querying makes classical uncertainty heuristics produce correlated and redundant selections.
- The proposed objective is therefore geometric coverage of the unlabeled pool rather than independent uncertainty ranking.
- The method solves each acquisition round myopically by selecting `b` new points conditioned on the existing labeled pool.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[sener-2018-active-1708-00489]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[sener-2018-active-1708-00489]].
