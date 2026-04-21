---
type: concept
title: Window Attention
slug: window-attention
date: 2026-04-20
updated: 2026-04-20
aliases: [sliding window attention]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Window attention** (窗口注意力) — an inference strategy that keeps only the most recent tokens' keys and values in cache, discarding older states to maintain bounded memory.

## Key Points

- Window attention gives constant-memory decoding after the cache fills, making it a natural candidate for streaming inference.
- The paper shows that it fails catastrophically once the first tokens are evicted, even if the retained recent context length remains large.
- On Llama-2-13B with cache `0 + 1024`, perplexity on a PG19 book explodes to `5158.07`.
- The failure is not because the oldest tokens are semantically crucial, but because evicting them removes the model's attention sinks.
- StreamingLLM can be understood as repairing window attention by preserving a tiny set of initial tokens alongside the rolling recent window.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[xiao-2024-efficient-2309-17453]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[xiao-2024-efficient-2309-17453]].
