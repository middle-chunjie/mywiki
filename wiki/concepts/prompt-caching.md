---
type: concept
title: Prompt Caching
slug: prompt-caching
date: 2026-04-20
updated: 2026-04-20
aliases: [prompt cache]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Prompt caching** — a serving strategy that precomputes and reuses intermediate attention state for prompt segments that recur across requests.

## Key Points

- In this paper, prompt caching is extended from root-level reuse to path-prefix reuse over the superposition DAG.
- The preamble KV cache and document KV caches are computed offline because they do not depend on the incoming query.
- The cached state is later combined with online query-path computation, reducing user-observed latency without changing model weights.
- The paper frames this as a lossless runtime optimization whose benefit compounds with path pruning and path parallelization.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[merth-2024-superposition-2404-06910]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[merth-2024-superposition-2404-06910]].
