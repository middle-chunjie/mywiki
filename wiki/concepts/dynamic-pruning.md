---
type: concept
title: Dynamic Pruning
slug: dynamic-pruning
date: 2026-04-20
updated: 2026-04-20
aliases: [动态剪枝]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Dynamic Pruning** (动态剪枝) — a retrieval-time strategy that skips or filters candidates adaptively based on partial scoring signals to reduce unnecessary ranking work.

## Key Points

- PLAID itself is built around progressive pruning, first at the centroid level and then at the document level before exact scoring.
- The paper explicitly interprets some PLAID cluster behavior as a possible form of dynamic pruning over lexical matches.
- The BM25 reranking baseline also benefits from dynamic index pruning in PISA, making small candidate pools much faster than large ones.
- The analysis shows pruning quality is highly parameter-dependent: aggressive pruning can remove useful candidates early with little latency benefit.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[macavaney-2024-reproducibility]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[macavaney-2024-reproducibility]].
