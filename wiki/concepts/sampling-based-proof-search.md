---
type: concept
title: Sampling-Based Proof Search
slug: sampling-based-proof-search
date: 2026-04-20
updated: 2026-04-20
aliases: [parallel proof sampling]
tags: [search, theorem-proving]
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Sampling-Based Proof Search** — a proof-generation strategy that samples multiple tactic trajectories in parallel and advances only through legal proof steps instead of maintaining a scored global search tree.

## Key Points

- [[unknown-nd-leanstar]] introduces this evaluation strategy because best-first search performs poorly when natural-language thoughts are interleaved with tactics.
- Each sample generates at most `N = 50` tactics at evaluation time, with `K = 32` or `64` parallel trajectories and temperature `T = 0.7`.
- Illegal tactics are discarded and resampled rather than inserted into an explicit search tree.
- The paper argues that the method is roughly comparable in compute to search with equal `S x K` budget, while handling hidden thought variables more naturally.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[unknown-nd-leanstar]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[unknown-nd-leanstar]].
