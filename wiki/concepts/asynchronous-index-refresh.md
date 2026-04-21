---
type: concept
title: Asynchronous Index Refresh
slug: asynchronous-index-refresh
date: 2026-04-20
updated: 2026-04-20
aliases: [异步索引刷新]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Asynchronous Index Refresh** (异步索引刷新) — a systems strategy that rebuilds a retrieval index in the background from a parameter snapshot while model training continues on the foreground process.

## Key Points

- REALM uses separate trainer and index-builder jobs so document embeddings can be recomputed without blocking gradient updates.
- The trainer sends a snapshot `θ'` to the index builder, which re-embeds and re-indexes the corpus before swapping in the new MIPS index.
- This design makes it practical to train a retriever whose document encoder is updated during pre-training.
- The analysis shows that slower refreshes lead to substantially worse QA accuracy and retrieval recall.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[guu-2020-realm-2002-08909]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[guu-2020-realm-2002-08909]].
