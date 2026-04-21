---
type: concept
title: Differentiable Search Index
slug: differentiable-search-index
date: 2026-04-20
updated: 2026-04-20
aliases: [DSI, 可微搜索索引]
tags: [retrieval, indexing]
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Differentiable Search Index** (可微搜索索引) — a generative retrieval formulation in which a sequence-to-sequence model stores corpus-to-identifier mappings in its parameters and retrieves by generating document ids directly from the query.

## Key Points

- The paper uses DSI as the baseline generative retrieval framework, with separate indexing and retrieval tasks sharing a seq2seq backbone.
- In DSI, the model must learn both document-content-to-docid and query-to-docid mappings, which exposes a mismatch between long indexing inputs and short retrieval queries.
- The paper shows that vanilla DSI degrades sharply with scale, especially when training relies only on labeled queries.
- The study argues that synthetic queries are the main practical fix within the DSI paradigm, while several decoder modifications add little benefit once compute is normalized.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[pradeep-2023-how]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[pradeep-2023-how]].
