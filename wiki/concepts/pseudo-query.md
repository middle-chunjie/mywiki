---
type: concept
title: Pseudo-Query
slug: pseudo-query
date: 2026-04-20
updated: 2026-04-20
aliases: [pseudo query, simulated query, 伪查询]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Pseudo-Query** (伪查询) — an automatically generated question-like representation attached to a passage or document unit to expose what information it can answer or what missing information it points to.

## Key Points

- HopRAG generates two pseudo-query sets per passage: in-coming questions answerable by the passage and out-coming questions that the passage raises but cannot answer alone.
- Each pseudo-query is converted into both keywords and dense vectors, so it can support hybrid matching during graph construction.
- Directed passage edges are formed by pairing an out-coming pseudo-query from one passage with the best-matching in-coming pseudo-query of another passage.
- During retrieval, the pseudo-query stored on an outgoing edge becomes the reasoning cue that helps the traversal model decide where to hop next.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[liu-2025-hoprag-2502-12442]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[liu-2025-hoprag-2502-12442]].
