---
type: concept
title: Singular Value Decomposition
slug: singular-value-decomposition
date: 2026-04-20
updated: 2026-04-20
aliases: [SVD, 奇异值分解]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Singular Value Decomposition** (奇异值分解) — a factorization `W = UΣV^T` that expresses a matrix through orthonormal singular vectors and ordered singular values, enabling optimal truncated low-rank approximation.

## Key Points

- LASER computes truncated SVD and keeps the top singular components when constructing the intervened matrix.
- The paper defines lower-order components as directions with larger singular values and higher-order components as directions with smaller singular values.
- Analyses using only higher-order components often produce generic tokens or semantically related but incorrect answers.
- SVD is used not only for compression, but also as a probe into how factual and noisy responses are distributed within model weights.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[sharma-2023-truth-2312-13558]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[sharma-2023-truth-2312-13558]].
