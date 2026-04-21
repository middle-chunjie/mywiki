---
type: concept
title: Dense Retrieval
slug: dense-retrieval
date: 2026-04-20
updated: 2026-04-20
aliases:
  - 稠密检索
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Dense Retrieval** (稠密检索) — a retrieval paradigm that embeds queries and documents into a shared vector space and ranks candidates by vector similarity.

## Key Points

- Retrieval training is asymmetric: the model prepends `Query:` to queries and `Document:` to candidate passages or documents.
- The retrieval adapter combines InfoNCE, embedding distillation, and GOR regularization rather than relying on a single retrieval loss.
- The small model reaches `63.28` average over five retrieval benchmark families, and the nano model reaches `61.43` while using only `239M` parameters.
- The paper emphasizes robustness for long documents, ANN search, and quantized deployment as part of the retrieval design.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[akram-2026-jinaembeddingsvtext-2602-15547]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[akram-2026-jinaembeddingsvtext-2602-15547]].
