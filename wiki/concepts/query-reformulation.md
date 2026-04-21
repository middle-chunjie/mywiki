---
type: concept
title: Query Reformulation
slug: query-reformulation
date: 2026-04-20
updated: 2026-04-20
aliases: [query rewriting, 查询重构]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Query Reformulation** (查询重构) — the transformation of an underspecified or context-dependent query into a more explicit retrieval query that better captures the user's current information need.

## Key Points

- HAConvDR reformulates the current conversational query with only PRJ-relevant historical turns instead of concatenating the full dialogue history.
- The reformulated query `` `q_n^r` `` may include both historical queries and historical gold passages, not just dialogue utterances.
- The paper positions denoised reformulation as an end-to-end retrieval-oriented alternative to relying purely on human-written conversational query rewriting.
- Removing PRJ-based reformulation causes the largest ablation drop on TopiOCQA, reducing `MRR` from `30.1` to `25.0`.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[mo-2024-historyaware-2401-16659]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[mo-2024-historyaware-2401-16659]].
