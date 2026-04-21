---
type: concept
title: Retrieval Precision
slug: retrieval-precision
date: 2026-04-20
updated: 2026-04-20
aliases: [precision of retrieved passages, 检索精确率]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Retrieval Precision** (检索精确率) — the proportion of retrieved passages that directly contain the true answer to a query.

## Key Points

- The paper uses retrieval precision as a concrete way to quantify imperfect retrieval under realistic web search.
- In the collected benchmark, low precision is common, with most retrieved passages failing to directly contain the correct answer.
- Conflict rates are highest when retrieval precision is very low, showing a direct link between noisy retrieval and knowledge conflict.
- Astute RAG improves performance across precision buckets and remains competitive even when precision is close to zero.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[wang-2024-astute-2410-07176]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[wang-2024-astute-2410-07176]].
