---
type: concept
title: Exact Lexical Match
slug: exact-lexical-match
date: 2026-04-20
updated: 2026-04-20
aliases: [exact match]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Exact Lexical Match** (精确词汇匹配) — a retrieval mechanism that scores query-document pairs only through terms that match in surface form, possibly after light normalization such as stemming.

## Key Points

- The paper treats exact lexical match as the efficiency-preserving core of inverted-index retrieval rather than an obsolete baseline to discard.
- Classical BM25 is presented as a special case where the contribution of each overlapping term is determined by `idf` and term-frequency statistics.
- COIL preserves the same exact-overlap constraint, but replaces frequency heuristics with contextualized token-vector similarity.
- The authors argue that exact matching offers better controllability and explainability than unconstrained soft matching over all token pairs.
- The empirical results suggest that exact lexical matching still has large headroom once contextual semantics are introduced.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[gao-2021-coil-2104-07186]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[gao-2021-coil-2104-07186]].
