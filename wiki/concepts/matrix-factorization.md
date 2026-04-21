---
type: concept
title: Matrix Factorization
slug: matrix-factorization
date: 2026-04-20
updated: 2026-04-20
aliases: [矩阵分解]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Matrix Factorization** (矩阵分解) — a modeling approach that approximates a matrix as the product of lower-dimensional latent factors to reconstruct observed entries and generalize to unobserved ones.

## Key Points

- This paper factorizes a partially observed query-item cross-encoder score matrix `G` into train-query embeddings `U` and item embeddings `V`.
- The optimization target is a sparse reconstruction loss over only observed entries rather than a dense matrix approximation.
- The learned factors are used to approximate cross-encoder relevance with inner products `u_q v_i^\top` for `k`-NN search.
- The method is positioned as a cheaper alternative to dense CUR-style decomposition and to full dual-encoder distillation.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[unknown-nd-adaptive-2405-03651]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[unknown-nd-adaptive-2405-03651]].
