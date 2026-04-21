---
type: concept
title: Proposition Aggregation
slug: proposition-aggregation
date: 2026-04-20
updated: 2026-04-20
aliases: [proposition aggregates, 命题聚合]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Proposition Aggregation** (命题聚合) — the construction of larger evidence units by grouping fine-grained propositions that share the same entity signal.

## Key Points

- [[unknown-nd-sirerag]] builds proposition aggregates by exact-match shared entities and keeps the original proposition order within each document.
- The aggregates act as pseudo-documents for the relatedness tree, making cross-document bridge facts retrievable without indexing every raw proposition.
- The paper deliberately excludes propositions with no associated entities to reduce noise from vague common nouns.
- Proposition aggregates are central to the method: removing them drops average F1 from `65.83` to `60.47`.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[unknown-nd-sirerag]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[unknown-nd-sirerag]].
