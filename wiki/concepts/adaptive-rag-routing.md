---
type: concept
title: Adaptive RAG Routing
slug: adaptive-rag-routing
date: 2026-04-20
updated: 2026-04-20
aliases: [adaptive RAG router, RAG routing]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Adaptive RAG Routing** — the task of selecting the most suitable retrieval-augmented generation paradigm for a specific query-corpus pair under an explicit utility objective.

## Key Points

- The paper formulates routing as `` `\pi^* = \arg\max_{\pi \in \Pi} \mathcal{U}(\pi; q, \mathcal{C})` ``, making paradigm choice a context-dependent optimization problem.
- Routing is driven by both online query signals and offline corpus fingerprints rather than query complexity alone.
- The benchmark treats effectiveness and efficiency jointly, so routing decisions can target answer quality, token budget, or a trade-off between them.
- The experiments show that optimal routing changes across both datasets and query types, ruling out a single universally best RAG strategy.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[wang-2026-ragrouterbench-2602-00296]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[wang-2026-ragrouterbench-2602-00296]].
