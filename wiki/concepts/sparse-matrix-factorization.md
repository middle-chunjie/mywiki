---
type: concept
title: Sparse Matrix Factorization
slug: sparse-matrix-factorization
date: 2026-04-20
updated: 2026-04-20
aliases: [稀疏矩阵分解]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Sparse Matrix Factorization** (稀疏矩阵分解) — matrix factorization performed on a partially observed matrix, where only a selected subset of entries is available and used during training.

## Key Points

- The paper builds a sparse score matrix by selecting only `k_d` items per train query or `k_d` queries per item for exact cross-encoder scoring.
- This sparse construction reduces offline indexing cost from dense `|Q_train||I|` scoring to a tractable number of cross-encoder calls.
- Factorization over observed entries is the mechanism that lets the method align item embeddings with the cross-encoder without dense supervision.
- The approach is central to the paper's claim that indexing can be much faster than adaCUR while retaining strong recall.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[unknown-nd-adaptive-2405-03651]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[unknown-nd-adaptive-2405-03651]].
