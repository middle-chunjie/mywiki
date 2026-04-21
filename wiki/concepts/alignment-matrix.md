---
type: concept
title: Alignment Matrix
slug: alignment-matrix
date: 2026-04-20
updated: 2026-04-20
aliases: [token alignment matrix, 对齐矩阵]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Alignment Matrix** (对齐矩阵) — a matrix that specifies or weights which query-document token pairs are allowed to interact when computing retrieval relevance.

## Key Points

- The paper treats the alignment matrix `A` as the central abstraction connecting generative retrieval and multi-vector dense retrieval.
- In MVDR, `A` is usually sparse and often computed heuristically, for example by selecting the best-matching document token for each query token.
- In GR, `A` is dense and learned through cross-attention, with rows normalized by softmax over query tokens for each predicted document token.
- The paper further argues that both MVDR and GR alignment matrices exhibit low-rank structure, making alignment a shared analytic lens rather than a paradigm-specific detail.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[wu-2024-generative-2404-00684]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[wu-2024-generative-2404-00684]].
