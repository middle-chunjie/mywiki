---
type: concept
title: Local backward window
slug: local-backward-window
date: 2026-04-20
updated: 2026-04-20
aliases: [局部后向窗口]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Local backward window** (局部后向窗口) — a small set of candidate documents immediately following the current generated item, used as a proxy for backward evidence when the full future ranking is still unknown.

## Key Points

- GenRT introduces this mechanism because truncation needs static-list evidence while reranking changes the list dynamically.
- At step `T`, the model selects the next `beta` items from the current local ranking as backward context for truncation.
- The paper sets ``beta = 4`` in experiments and shows this value balances useful future evidence against local-ranking noise.
- Performance deteriorates when the window grows too large, which the authors attribute to noisy backward approximations.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[xu-2024-listaware-2402-02764]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[xu-2024-listaware-2402-02764]].
