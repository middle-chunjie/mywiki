---
type: concept
title: Retrieval Granularity
slug: retrieval-granularity
date: 2026-04-20
updated: 2026-04-20
aliases: [retrieval unit granularity, granularity choice, 检索粒度]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Retrieval granularity** (检索粒度) — the unit size used to segment and index a corpus for retrieval, such as documents, passages, sentences, or propositions.

## Key Points

- The paper treats granularity as an inference-time decision rather than something fixed by retriever training.
- Passage, sentence, and proposition indexing expose a trade-off between context coverage and information density.
- Finer granularity can improve recall even when retrievers were trained only on passage-level supervision.
- Granularity also changes downstream prompt construction because the same token budget can hold more atomic facts.
- The reported gains are largest on cross-task settings and questions about long-tail entities.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[chen-2024-dense-2312-06648]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[chen-2024-dense-2312-06648]].
