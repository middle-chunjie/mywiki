---
type: concept
title: Sliding-Window Recomputation
slug: sliding-window-recomputation
date: 2026-04-20
updated: 2026-04-20
aliases: [sliding window with recomputation]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Sliding-window recomputation** — a streaming baseline that repeatedly rebuilds key-value states for the recent context window at each decoding step instead of storing the full history.

## Key Points

- The paper treats this baseline as the only practical oracle with strong perplexity on long streams.
- It preserves good language-modeling quality because recent states are re-derived from a contiguous window at every step.
- Its cost grows quadratically with the window length inside each refresh, making it much slower than cache reuse.
- StreamingLLM nearly matches its perplexity while achieving up to `22.2x` lower per-token decoding latency.
- Memory usage is similar to the recomputation baseline, so the main gain comes from eliminating repeated quadratic attention work.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[xiao-2024-efficient-2309-17453]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[xiao-2024-efficient-2309-17453]].
