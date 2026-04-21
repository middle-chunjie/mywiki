---
type: concept
title: Multi-Task Learning
slug: multi-task-learning
date: 2026-04-20
updated: 2026-04-20
aliases: [多任务学习, joint training]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Multi-Task Learning** (多任务学习) — training a shared model on multiple related objectives so that supervision from one task can improve generalization on others.

## Key Points

- The paper studies combinations such as `RFL`, `RFLH`, `RFLHU`, and `HU` over BLANCA tasks.
- Multi-task training usually helps forum-oriented tasks more than single-task fine-tuning.
- Transfer is asymmetric: hierarchy and usage supervision help forum tasks more than forum tasks help hierarchy or usage.
- These results suggest code-property tasks and forum-text tasks emphasize partly different features.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[abdelaziz-2022-can]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[abdelaziz-2022-can]].
