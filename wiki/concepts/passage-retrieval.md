---
type: concept
title: Passage Retrieval
slug: passage-retrieval
date: 2026-04-20
updated: 2026-04-20
aliases: [passage-level retrieval, 段落检索]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Passage retrieval** (段落检索) — retrieving and ranking passages as the target evidence units for a query, either directly or by aggregating scores from finer-grained subunits.

## Key Points

- Passage retrieval is the primary evaluation target in this paper even when the underlying index uses sentences or propositions.
- For fine-grained indexes, the paper maps retrieved units back to source passages and keeps the top-`k` unique passages.
- Passage-level indexing provides broader context but often mixes relevant facts with distracting details.
- Higher passage recall in the experiments correlates closely with higher downstream QA accuracy from a reader model.
- The paper shows that passage retrieval can be improved by finer indexing without retraining the retriever on new supervision.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[chen-2024-dense-2312-06648]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[chen-2024-dense-2312-06648]].
