---
type: concept
title: Contextualized Inverted List
slug: contextualized-inverted-list
date: 2026-04-20
updated: 2026-04-20
aliases: []
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Contextualized Inverted List** — an inverted-list index that stores contextualized token vectors instead of only term statistics, enabling exact lexical lookup with semantic-aware scoring.

## Key Points

- COIL builds one list `I^t` per vocabulary token `t`, containing all contextualized document-token vectors whose surface form is `t`.
- Query-time retrieval touches only the subset of lists corresponding to query tokens, preserving the selective-access property of classical lexical retrieval.
- Each list is implemented as a dense matrix so the matching computation becomes a matrix-vector product over all mentions of that token.
- The structure replaces term-frequency-based posting payloads with BERT-derived token representations while keeping document IDs for score aggregation.
- The design is central to COIL's claim that exact-match retrieval can gain semantic expressivity without adopting a single monolithic dense index.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[gao-2021-coil-2104-07186]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[gao-2021-coil-2104-07186]].
