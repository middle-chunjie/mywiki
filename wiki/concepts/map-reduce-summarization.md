---
type: concept
title: Map-Reduce Summarization
slug: map-reduce-summarization
date: 2026-04-20
updated: 2026-04-20
aliases: [map reduce summarization, hierarchical summarization]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Map-Reduce Summarization** — a summarization strategy that first produces multiple intermediate summaries or answers in parallel and then combines them into a final consolidated summary.

## Key Points

- GraphRAG uses map-reduce both conceptually and operationally for global answer generation over community summaries.
- In the map stage, partial answers are generated in parallel and assigned helpfulness scores in the range `0-100`.
- In the reduce stage, the highest-scoring partial answers are sorted, packed into a new context, and summarized into the final answer.
- The paper also uses the same map-reduce procedure directly over source texts as the `TS` baseline.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[edge-2024-local-2404-16130]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[edge-2024-local-2404-16130]].
