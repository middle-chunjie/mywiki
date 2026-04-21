---
type: concept
title: Kernel Pooling
slug: kernel-pooling
date: 2026-04-20
updated: 2026-04-20
aliases: [核池化, kernel-based pooling]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Kernel Pooling** (核池化) — a differentiable pooling mechanism that summarizes interaction scores by counting how often they fall near multiple kernel centers.

## Key Points

- K-NRM applies `11` RBF kernels over query-document cosine similarities to produce multi-level soft-TF features.
- One narrow kernel at `μ = 1.0` captures exact matches, while the remaining kernels cover softer similarity bands from `0.9` to `-0.9`.
- Because the kernels are differentiable, ranking gradients can flow back into token similarities and word embeddings.
- The paper shows kernel pooling is materially better than max-pooling or mean-pooling variants for ad-hoc ranking.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[xiong-2017-endtoend-1706-06613]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[xiong-2017-endtoend-1706-06613]].
