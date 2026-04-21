---
type: concept
title: Vector Database
slug: vector-database
date: 2026-04-20
updated: 2026-04-20
aliases: [vector database, 向量数据库]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Vector Database** (向量数据库) — a storage and retrieval system optimized for embedding vectors and nearest-neighbor search over semantic representations.

## Key Points

- The base Mem0 pipeline uses a dense vector database to retrieve semantically similar existing memories during updates.
- This retrieval stage supplies the comparison set that the LLM uses to decide `ADD`, `UPDATE`, `DELETE`, or `NOOP`.
- Using a vector database allows the system to compare compact memory items rather than scanning entire conversation transcripts.
- The paper positions vector retrieval as a key efficiency mechanism behind low search latency.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[chhikara-nd-mem]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[chhikara-nd-mem]].
