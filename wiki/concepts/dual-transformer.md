---
type: concept
title: Dual Transformer
slug: dual-transformer
date: 2026-04-20
updated: 2026-04-20
aliases: [双 Transformer, dual-transformer module]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Dual Transformer** (双 Transformer) — a parallel-transformer architecture that processes two correlated sequences separately, here item IDs and item categories, to preserve distinct preference dynamics.

## Key Points

- HPM instantiates one self-attention stack for item IDs and another for category sequences instead of fusing them early.
- Both branches use position embeddings, multi-head self-attention, feed-forward layers, residual connections, dropout, and layer normalization.
- The final user representations are pooled independently as `v_f` and `c_f`, which feed both ranking and contrastive objectives.
- An ablation replacing the dual structure with one single transformer hurts performance across all tested datasets.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[huang-2023-dual]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[huang-2023-dual]].
