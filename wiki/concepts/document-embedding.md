---
type: concept
title: Document Embedding
slug: document-embedding
date: 2026-04-20
updated: 2026-04-20
aliases:
  - 文档嵌入
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Document Embedding** (文档嵌入) — a vector representation of a document used to support similarity search, retrieval, or downstream prediction in a shared latent space.

## Key Points

- The paper argues that conventional document embeddings are usually learned as `phi(d)`, depending only on the document itself and not on the target corpus.
- This lack of corpus awareness makes dense embeddings brittle under domain shift, especially when token salience changes between training and deployment corpora.
- CDE reframes document embeddings as potentially contextual objects `phi(d; D)`, borrowing the intuition of corpus-dependent weighting from sparse retrieval.
- The proposed architecture preserves the standard retrieval interface by still outputting one final embedding per document, so ANN indexing does not need to change.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[morris-2024-contextual-2410-02525]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[morris-2024-contextual-2410-02525]].
