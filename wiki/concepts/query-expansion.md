---
type: concept
title: Query Expansion
slug: query-expansion
date: 2026-04-20
updated: 2026-04-20
aliases: [query augmentation, 查询扩展]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Query Expansion** (查询扩展) — a retrieval technique that appends or rewrites a query with additional terms or context to improve recall of relevant documents.

## Key Points

- The paper implements query expansion by concatenating the original question with the pseudo-document generated in the previous iteration.
- This expansion is designed to bridge semantic gaps that prevent a retriever from finding the correct evidence from a short question alone.
- The expanded query is used with a dense retriever rather than sparse lexical matching.
- In ITRG, query expansion is iterative, so later retrieval steps depend on the quality of earlier generations.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[feng-2023-retrievalgeneration-2310-05149]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[feng-2023-retrievalgeneration-2310-05149]].
