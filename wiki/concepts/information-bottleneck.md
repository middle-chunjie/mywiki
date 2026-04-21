---
type: concept
title: Information Bottleneck
slug: information-bottleneck
date: 2026-04-20
updated: 2026-04-20
aliases: [IB, 信息瓶颈]
tags: [information-theory, retrieval]
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Information Bottleneck** (信息瓶颈) — a principle for learning a representation that compresses an input while retaining as much task-relevant information as possible about a target variable.

## Key Points

- The paper formulates GDR indexing as minimizing `I(X;T)` while preserving retrieval-relevant information through `I(T;Q)`.
- Under `T \leftrightarrow X \leftrightarrow Q`, the objective can be rewritten as `I(X;T) + \beta I(X;Q|T) + const`, making distortion explicit.
- Empirical bottleneck curves over `I(X;T)` and `I(X;Q|T)` are used to compare indexing methods beyond standard Rec@N scores.
- BMI produces a better bottleneck curve than hierarchical random indexing, LSH, and document-space `k`-means.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[du-2024-bottleneckminimal-2405-10974]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[du-2024-bottleneckminimal-2405-10974]].
