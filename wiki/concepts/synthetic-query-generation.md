---
type: concept
title: Synthetic Query Generation
slug: synthetic-query-generation
date: 2026-04-20
updated: 2026-04-20
aliases: [synthetic queries, document-to-query generation, 合成查询生成]
tags: [retrieval, data-augmentation]
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Synthetic Query Generation** (合成查询生成) — the practice of generating pseudo-queries for documents or passages so retrieval models can train on query-like inputs even when human-labeled queries are sparse.

## Key Points

- The paper uses docT5query to generate `40` synthetic queries per MS MARCO passage and reuses prior generated queries for NQ100k and TriviaQA.
- Synthetic queries address both the coverage gap and the distribution gap between indexing inputs and real retrieval queries.
- On MS MARCO, synthetic queries are the dominant factor behind scaling generative retrieval, yielding roughly `3x` gains over labeled-query-only DSI variants.
- More synthetic-query diversity helps: using all `100` sampled queries per passage improves MSMarco100k from `80.3` to `82.4` MRR@10.
- In-domain query generation matters on small corpora, with in-domain D2Q outperforming prior query generators on NQ100k and TriviaQA.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[pradeep-2023-how]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[pradeep-2023-how]].
