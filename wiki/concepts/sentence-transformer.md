---
type: concept
title: Sentence Transformer
slug: sentence-transformer
date: 2026-04-20
updated: 2026-04-20
aliases: [sentence transformer]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Sentence Transformer** — an encoder model that maps text or code into dense vectors so semantic similarity can be computed with simple geometric operations such as cosine similarity.

## Key Points

- [[geng-2024-large]] uses the sentence-transformer model `st-codesearch-distilroberta-base` to embed code snippets for semantic retrieval.
- The same semantic-vector idea is reused to compare generated comments during reranking.
- In the paper, semantic retrieval usually outperforms random demonstration selection and is strongest in the `10`-shot setting.
- The method operationalizes code similarity beyond surface token overlap.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[geng-2024-large]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[geng-2024-large]].
