---
type: concept
title: Position ID Reordering
slug: position-id-reordering
date: 2026-04-20
updated: 2026-04-20
aliases: [reordered positions, position rearrangement, 位置编号重排]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Position ID Reordering** (位置编号重排) — a technique that rewrites per-chunk position indices into one monotone global sequence before online KV-cache concatenation so relative-position encodings remain consistent with the assembled context.

## Key Points

- TurboRAG identifies composite positions like `[0, ..., l, 0, ..., l]` as a source of positional inconsistency when separately cached chunks are concatenated.
- The reordered variant assigns positions as `[0, ..., l, l+1, ..., 2l, ...]`, preserving the intended relative offsets between the query and each cached chunk.
- The paper relies on RoPE's dependence on relative position differences to justify why reordered positions better preserve attention behavior.
- Empirically, TurboRAG-reordered outperforms TurboRAG-composite on both RGB and LongBench, making this positional fix a central part of the method.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[unknown-nd-turborag]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[unknown-nd-turborag]].
