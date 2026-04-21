---
type: concept
title: Unit-Norm Embedding
slug: unit-norm-embedding
date: 2026-04-20
updated: 2026-04-20
aliases: [单位范数嵌入, unit-normalized embedding, l2-normalized embedding]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Unit-Norm Embedding** (单位范数嵌入) — an embedding vector normalized to `||x||_2 = 1`, so retrieval comparisons depend primarily on angular similarity.

## Key Points

- The paper's compression method assumes embeddings lie on the unit hypersphere, allowing the radius term to be omitted entirely.
- This geometric constraint is what causes the spherical angles to concentrate and become compressible.
- The method is evaluated on text, image, and multi-vector embeddings that are explicitly unit-normalized.
- The paper argues that the same assumption aligns naturally with cosine-similarity-based retrieval pipelines.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[xiao-2026-embedding-2602-00079]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[xiao-2026-embedding-2602-00079]].
