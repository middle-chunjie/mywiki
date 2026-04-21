---
type: concept
title: Jaccard Similarity
slug: jaccard-similarity
date: 2026-04-20
updated: 2026-04-20
aliases: [Jaccard coefficient, Jaccard index, Jaccard 相似度]
tags: []
source_count: 1
confidence: low
domain_volatility: low
last_reviewed: 2026-04-20
---

## Definition

**Jaccard Similarity** (Jaccard 相似度) — a set-overlap measure defined as the size of the intersection divided by the size of the union of two token sets.

## Key Points

- [[geng-2024-large]] uses Jaccard similarity to rank candidate code demonstrations by lexical overlap with a target method.
- Before scoring, the paper removes Java keywords, splits identifiers into subtokens, and lowercases them.
- The score is computed as `|tokens_target ∩ tokens_candidate| / |tokens_target ∪ tokens_candidate|`.
- This token-based retrieval strategy substantially improves prompting quality over random example selection.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[geng-2024-large]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[geng-2024-large]].
