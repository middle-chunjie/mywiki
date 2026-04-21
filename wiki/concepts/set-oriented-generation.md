---
type: concept
title: Set-Oriented Generation
slug: set-oriented-generation
date: 2026-04-20
updated: 2026-04-20
aliases: [set-based generation, unordered identifier generation, 集合导向生成]
tags: [generative-retrieval, decoding, document-identifier]
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Set-Oriented Generation** (集合导向生成) — a decoding paradigm for auto-regressive search engines in which the model generates the constituent terms of an unordered document identifier in any valid order, rather than being required to reproduce a single fixed token sequence.

## Key Points

- The key relaxation: a document is retrieved if *any* permutation of its term-set `T(D)` is generated; relevance is scored by the maximum likelihood across all generated permutations: `Rel(Q, D) = max{...}`.
- This directly addresses the false-pruning problem of sequential DocIDs, where one wrong step at any position in the sequence causes the document to be irrecoverably lost from the beam.
- At inference time, set-oriented generation is realized by constrained greedy search: a validity constraint prevents selecting already-used terms or terms that would lead to no valid document, maintained via an inverted-index data structure updated incrementally.
- Ablation comparing term-set vs. sequence-based identifiers (terms ordered by importance) shows gains of 0.024 MRR@10 and 0.027 Recall@10 on NQ320K, confirming that order relaxation is the key improvement beyond better term selection alone.
- The flexibility also allows the model to surface query-adaptive orderings: the term that is most likely given the query is decoded first, making early decoding steps easier and improving recall for unseen documents.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[zhang-2024-generative-2305-13859]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[zhang-2024-generative-2305-13859]].
