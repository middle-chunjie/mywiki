---
type: concept
title: Similarity Function
slug: similarity-function
date: 2026-04-20
updated: 2026-04-20
aliases: [similarity measure, 相似度函数]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Similarity Function** (相似度函数) — a function `k(x, x')` that quantifies how alike two items are under a chosen representation and thereby defines the notion of diversity being measured.

## Key Points

- Vendi Score delegates the semantics of diversity to the similarity function instead of fixing one global representation.
- The paper uses different similarities in different domains, including Morgan fingerprints, Inception cosine similarity, n-gram overlap, and probability-product kernels.
- The authors stress that the quality of the diversity estimate depends heavily on whether the chosen similarity function matches the application.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[friedman-2023-vendi-2210-02410]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[friedman-2023-vendi-2210-02410]].
