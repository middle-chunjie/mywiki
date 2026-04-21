---
type: concept
title: Query Difficulty Annotation
slug: query-difficulty-annotation
date: 2026-04-20
updated: 2026-04-20
aliases: [difficulty labeling, 查询难度标注]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Query Difficulty Annotation** (查询难度标注) — the practice of assigning categorical difficulty labels to retrieval queries based on nuisance factors that affect matchability.

## Key Points

- [[wu-2023-forb-2309-16249]] labels each query as `easy`, `medium`, or `hard`.
- The labels are based on seven factors: occlusion, blur, truncation, color distortion, perspective distortion, texture complexity, and target object area.
- Multiple annotators label the queries and majority voting determines the final difficulty category.
- The resulting labels correlate well with retrieval performance across methods, making the benchmark more diagnostic than a single aggregate score.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[wu-2023-forb-2309-16249]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[wu-2023-forb-2309-16249]].
