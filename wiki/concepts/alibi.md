---
type: concept
title: ALiBi
slug: alibi
date: 2026-04-20
updated: 2026-04-20
aliases: [Attention with Linear Biases]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**ALiBi** — a relative-position method that adds distance-dependent linear biases to attention scores instead of using explicit positional embeddings.

## Key Points

- The paper uses MPT as its main ALiBi-based model family when testing StreamingLLM across positional-encoding schemes.
- StreamingLLM remains compatible with ALiBi by applying a contiguous linear bias inside the current cache rather than the bias implied by the original text positions.
- This cache-local position handling is described as necessary for stable streaming behavior.
- On MPT-7B, pure window attention reaches perplexity `460.29`, while StreamingLLM with four sink tokens restores it to about `14.99`.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[xiao-2024-efficient-2309-17453]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[xiao-2024-efficient-2309-17453]].
