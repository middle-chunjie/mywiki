---
type: concept
title: Query-Corpus Compatibility
slug: query-corpus-compatibility
date: 2026-04-20
updated: 2026-04-20
aliases: [query corpus compatibility]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Query-Corpus Compatibility** — the degree to which a query’s information demands align with the structural and semantic properties of the target corpus, thereby affecting which retrieval paradigm works best.

## Key Points

- The paper argues that routing errors arise when systems model only query difficulty and ignore corpus structure or semantic geometry.
- Structural metrics such as LCC ratio, density, and clustering coefficient are used to estimate how well graph-based retrieval can operate on a corpus.
- Semantic metrics such as intrinsic dimension, dispersion, and hubness are used to explain when dense retrieval is likely to suffer from crowding or interference.
- The benchmark’s core empirical claim is that paradigm performance reversals are explained by this query-corpus interaction rather than by query type alone.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[wang-2026-ragrouterbench-2602-00296]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[wang-2026-ragrouterbench-2602-00296]].
