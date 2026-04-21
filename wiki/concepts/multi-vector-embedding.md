---
type: concept
title: Multi-Vector Embedding
slug: multi-vector-embedding
date: 2026-04-20
updated: 2026-04-20
aliases: [多向量嵌入]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Multi-Vector Embedding** (多向量嵌入) — an embedding representation in which one input is encoded as multiple vectors, typically one per token or local unit, instead of a single dense vector.

## Key Points

- The paper highlights multi-vector retrieval as a major storage stress case because each document may contribute dozens of embeddings.
- `ColBERT`-style late interaction is the motivating example, with around `50` to `100` embeddings per document in the paper's discussion.
- The proposed compressor is shown to retain gains on multi-vector settings, including `jina-colbert-v2`.
- The paper estimates that improved compression can save tens of gigabytes to hundreds of gigabytes at realistic retrieval scale.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[xiao-2026-embedding-2602-00079]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[xiao-2026-embedding-2602-00079]].
