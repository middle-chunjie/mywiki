---
type: concept
title: Blockwise Token Selection
slug: blockwise-token-selection
date: 2026-04-20
updated: 2026-04-20
aliases: [blockwise selection, 块级令牌选择]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Blockwise Token Selection** (块级令牌选择) — a sparse attention strategy that scores and retains contiguous token blocks rather than isolated tokens so computation and memory access remain hardware-efficient.

## Key Points

- [[yuan-2025-native-2502-11089]] selects sparse context in continuous blocks because contiguous KV loads are more GPU-friendly than random token fetches.
- NSA derives block importance from compression-branch attention scores instead of running a separate expensive selector over all original tokens.
- The method aggregates block scores across heads within each GQA group so shared KV caches can be reused efficiently during decoding.
- In the main configuration, NSA uses selected block size ``l' = 64`` and retains top-``n = 16`` blocks per query.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[yuan-2025-native-2502-11089]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[yuan-2025-native-2502-11089]].
