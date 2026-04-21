---
type: concept
title: Vector RAG
slug: vector-rag
date: 2026-04-20
updated: 2026-04-20
aliases: [semantic search rag, vector retrieval rag]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Vector RAG** — a retrieval-augmented generation setup that selects semantically similar chunks in vector space and conditions answer generation on those retrieved chunks.

## Key Points

- The paper uses vector RAG as the conventional baseline and labels it `SS` for semantic search.
- Vector RAG works well for localized questions but fails on global corpus questions that require aggregation across many records.
- Across both datasets, vector RAG loses substantially on comprehensiveness and diversity to GraphRAG-style global methods.
- The baseline remains strongest on directness because retrieved local passages encourage shorter, more targeted answers.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[edge-2024-local-2404-16130]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[edge-2024-local-2404-16130]].
