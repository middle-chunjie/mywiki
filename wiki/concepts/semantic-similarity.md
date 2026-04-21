---
type: concept
title: Semantic Similarity
slug: semantic-similarity
date: 2026-04-20
updated: 2026-04-20
aliases: [semantic similarity, 语义相似性]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Semantic Similarity** (语义相似性) — the degree to which two representations express similar meaning, often computed from embeddings rather than surface-form overlap.

## Key Points

- Mem0 retrieves the top similar memories for each candidate fact using dense semantic similarity search.
- In `Mem0^g`, semantic similarity is also used to decide whether a new entity should match an existing node or create a new one.
- The graph retrieval module ranks relation triples against dense query embeddings to find relevant memory paths.
- The paper relies on semantic similarity to avoid redundant memory creation and to keep retrieval targeted.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[chhikara-nd-mem]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[chhikara-nd-mem]].
