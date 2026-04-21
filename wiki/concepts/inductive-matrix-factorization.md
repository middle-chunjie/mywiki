---
type: concept
title: Inductive Matrix Factorization
slug: inductive-matrix-factorization
date: 2026-04-20
updated: 2026-04-20
aliases: [归纳式矩阵分解]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Inductive Matrix Factorization** (归纳式矩阵分解) — a factorization method that uses input features and parametric encoders so it can produce latent representations for previously unseen rows or columns.

## Key Points

- In this paper, `MFInd` learns shallow MLPs on top of frozen `DE_src` embeddings for both queries and items.
- The model uses a 2-layer GELU MLP with a learned skip connection, which the authors report generalizes better than a plain MLP without the skip path.
- Because it is inductive, the method can embed unseen test queries and items not directly scored in the sparse matrix `G`.
- The paper shows `MFInd` is especially useful on large-scale domains such as Hotpot-QA, where transductive item-parameter training becomes too sparse or too expensive.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[unknown-nd-adaptive-2405-03651]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[unknown-nd-adaptive-2405-03651]].
