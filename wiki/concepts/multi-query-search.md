---
type: concept
title: Multi-Query Search
slug: multi-query-search
date: 2026-04-20
updated: 2026-04-20
aliases: [parallel query generation, 多查询搜索]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Multi-Query Search** (多查询搜索) — a retrieval strategy that issues several diverse queries in parallel for the same reasoning step so the system can cover complementary evidence more efficiently.

## Key Points

- InfoSeeker generates multiple search queries at each reasoning turn instead of relying on a single sequential query.
- The paper motivates this design as a way to increase recall and exploration breadth without letting a long retrieval context overwhelm the model.
- Retrieved results are summarized by a Refiner Agent, which compresses top-`k` documents into concise evidence blocks tied to the originating query.
- The approach is positioned as a core inference-time workflow choice behind InfoSeeker's deep-research behavior.
- Multi-query search is paired with explicit thinking steps so the model can decide which information gaps to probe next.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[xia-2025-open-2509-00375]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[xia-2025-open-2509-00375]].
