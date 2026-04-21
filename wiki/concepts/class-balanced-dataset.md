---
type: concept
title: Class-Balanced Dataset
slug: class-balanced-dataset
date: 2026-04-20
updated: 2026-04-20
aliases: [balanced dataset, 类别均衡数据集]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Class-Balanced Dataset** (类别均衡数据集) — a labeled dataset designed so each class has comparable representation, reducing skew in aggregate evaluation.

## Key Points

- SORRY-Bench uses exactly `10` unsafe instructions for each of its `44` risk categories.
- The balanced construction is motivated by heavy over-representation of a few harm types in prior safety datasets.
- The authors combine filtered legacy examples with newly authored prompts to fill under-covered classes.
- This balance lets benchmark aggregates reflect cross-category safety more faithfully instead of mirroring prior data skew.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[xie-2024-sorrybench-2406-14598]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[xie-2024-sorrybench-2406-14598]].
