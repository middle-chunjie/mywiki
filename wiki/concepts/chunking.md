---
type: concept
title: Chunking
slug: chunking
date: 2026-04-20
updated: 2026-04-20
aliases: [Chunking, 分块]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Chunking** (分块) — the strategy of splitting a long sequence into smaller segments so a model can process it under limited context length or compute budgets.

## Key Points

- [[li-2024-chulo-2410-11119]] uses fixed-length non-overlapping chunks with padding for incomplete final segments.
- The paper treats chunk size `n` as a key tradeoff parameter controlling compression versus information granularity.
- ChuLo argues that naive chunking can fragment semantics, so chunking is paired with document-level keyphrase extraction and weighted aggregation.
- Optimal chunk sizes differ by dataset, from `5` on EURLEX variants to `50` on LUN and GUM.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[li-2024-chulo-2410-11119]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[li-2024-chulo-2410-11119]].
