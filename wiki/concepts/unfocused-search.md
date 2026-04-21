---
type: concept
title: Unfocused Search
slug: unfocused-search
date: 2026-04-20
updated: 2026-04-20
aliases: [generic search, diffuse querying]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Unfocused Search** — a failure mode where an agent issues overly generic search queries that do not narrow the search space or advance evidence gathering.

## Key Points

- [[yen-2025-lost-2510-18939]] defines unfocused search over the whole trajectory, not a single bad query.
- The paper marks a trajectory as unfocused when a majority of its searches are too generic to make progress.
- Detection is performed with an LLM judge over the collected search queries.
- Slim reduces unfocused search relative to HF-ODR (`34.0` versus `58.7`) but remains close to Search-o1 (`33.7`).
- The separation of search snippets from targeted browsing is intended to make later queries more focused.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[yen-2025-lost-2510-18939]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[yen-2025-lost-2510-18939]].
