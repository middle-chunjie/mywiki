---
type: concept
title: Retrieval Query Length
slug: retrieval-query-length
date: 2026-04-20
updated: 2026-04-20
aliases: [query length for retrieval, 检索查询长度]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Retrieval query length** (检索查询长度) — the number of recent prefix tokens used to form the retrieval query in retrieval-augmented generation or language modeling.

## Key Points

- [[ram-2023-incontext-2302-00083]] denotes retrieval query length as `l`, restricting the query to the last `l` tokens rather than the full prefix.
- Query length controls a tradeoff between contextualization and recency: too short omits useful context, while too long dilutes the most local signal.
- In the paper, BM25 performs best at `l = 32`, whereas dense retrievers prefer `l = 64`.
- The authors show that tying query length to retrieval stride, as some prior systems do, can hurt language modeling quality.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[ram-2023-incontext-2302-00083]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[ram-2023-incontext-2302-00083]].
