---
type: concept
title: Query Generation
slug: query-generation
date: 2026-04-20
updated: 2026-04-20
aliases: [search query generation, query drafting]
tags: [retrieval, reasoning]
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Query Generation** (查询生成) — the process of producing a search query that exposes the right external evidence for the current reasoning state.

## Key Points

- CR-Planner treats query generation as its own sub-goal instead of folding it implicitly into one-shot prompting.
- The paper argues that flawed intermediate reasoning can produce misleading search queries, which then amplify retrieval errors.
- A dedicated query critic scores alternative query candidates given the immediately preceding rationale.
- This design is intended to make retrieval responsive to the current reasoning context rather than only to the original problem statement.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[li-2024-can-2410-01428]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[li-2024-can-2410-01428]].
