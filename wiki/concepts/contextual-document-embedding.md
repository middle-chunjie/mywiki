---
type: concept
title: Contextual Document Embedding
slug: contextual-document-embedding
date: 2026-04-20
updated: 2026-04-20
aliases:
  - 上下文化文档嵌入
  - CDE
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Contextual Document Embedding** (上下文化文档嵌入) — a document embedding that conditions its representation not only on the target text itself but also on neighboring or corpus-level contextual documents.

## Key Points

- The paper's central proposal is to encode sampled corpus documents with a first-stage model `M_1` and inject them into a second-stage encoder `M_2` when embedding a target document or query.
- This design aims to recover corpus-sensitive signals analogous to IDF while keeping the final output a fixed-size dense vector for standard retrieval.
- Context can be shared across a training batch and cached at indexing time, so contextualization does not require storing extra vectors per document at serving time.
- The learned embedding changes as its conditioning corpus changes, and the paper shows that true in-domain context outperforms random substitute context on MTEB.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[morris-2024-contextual-2410-02525]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[morris-2024-contextual-2410-02525]].
