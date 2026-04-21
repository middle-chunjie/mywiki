---
type: concept
title: Bridge Attention
slug: bridge-attention
date: 2026-04-20
updated: 2026-04-20
aliases: [bridge-token attention]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Bridge Attention** — a sparse attention mechanism that inserts special bridge tokens to summarize local code spans and make their information globally reachable through a small number of attention hops.

## Key Points

- LongCoder inserts `m` bridge tokens roughly uniformly across the input sequence.
- Each bridge token reads from its preceding chunk of about `ceil(n / m)` tokens, while later tokens can attend to the bridge token directly.
- This reduces the path length to distant earlier context to at most `2` hops, compared with repeated propagation through local windows.
- The ablation removing bridge tokens causes the largest drop in Edit Similarity, showing that the summarized span representation matters for completion quality.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[guo-2023-longcoder-2306-14893]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[guo-2023-longcoder-2306-14893]].
