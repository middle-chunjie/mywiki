---
type: concept
title: Transductive Matrix Factorization
slug: transductive-matrix-factorization
date: 2026-04-20
updated: 2026-04-20
aliases: [传导式矩阵分解]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Transductive Matrix Factorization** (传导式矩阵分解) — a factorization setting in which latent representations for the observed rows and columns are learned directly as parameters, without a parametric encoder for unseen instances.

## Key Points

- The paper's `MFTrns` directly optimizes train-query embeddings `U` and item embeddings `V` as free parameters.
- It can optionally initialize those embeddings from `DE_src`, then refine them using the sparse cross-encoder score matrix.
- This setting works well on smaller domains because each item is observed often enough to update its embedding reliably.
- On very large corpora, the approach scales poorly because the number of trainable item parameters grows linearly with the corpus size.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[unknown-nd-adaptive-2405-03651]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[unknown-nd-adaptive-2405-03651]].
