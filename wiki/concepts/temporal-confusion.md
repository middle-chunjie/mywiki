---
type: concept
title: Temporal Confusion
slug: temporal-confusion
date: 2026-04-20
updated: 2026-04-20
aliases: [position collision in cache]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Temporal Confusion** (时间混淆) — the positional inconsistency that arises when cached suffix tokens retain pre-edit indices after an insertion or deletion, causing attention to compare against keys with incorrect or duplicated positions.

## Key Points

- The paper identifies temporal confusion as the main structural failure mode of directly splicing edited KV entries into an existing RoPE-based cache.
- The issue appears when `j - i != m`, so the edited sequence length differs from the replaced span and suffix positions shift.
- In the authors' code-editing setting, resolving temporal confusion alone is usually enough to recover nearly all of the accuracy lost by naive conflict fast encoding.
- PIE fixes the problem by removing stale rotary matrices from suffix keys and reapplying the correct post-edit positional rotation.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[he-2024-let-2407-03157]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[he-2024-let-2407-03157]].
