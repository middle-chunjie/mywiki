---
type: concept
title: Phrase Retrieval
slug: phrase-retrieval
date: 2026-04-20
updated: 2026-04-20
aliases: [dense phrase retrieval, 短语检索]
tags: [retrieval, qa]
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Phrase Retrieval** (短语检索) — a retrieval paradigm that indexes and scores contiguous text spans rather than whole passages or documents, enabling answer-span-level search via dense similarity.

## Key Points

- The paper treats any contiguous span up to `L = 20` words as a retrievable phrase and scores it with a dense inner product against the query.
- Passage retrieval can be derived from phrase retrieval by taking the maximum score over phrases inside a passage, so phrase indexing becomes a multi-granularity retrieval substrate.
- In-passage negatives provide stronger fine-grained supervision than passage-level hard negatives because the negatives share both topic and local context with the positive answer phrase.
- Query-side fine-tuning with document-level supervision extends phrase retrieval from answer extraction to coarser retrieval tasks such as entity linking and grounded dialogue.
- Phrase filtering plus OPQ compression makes the approach practical by reducing the number of stored vectors and the size of the phrase index.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[lee-2021-phrase-2109-08133]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[lee-2021-phrase-2109-08133]].
