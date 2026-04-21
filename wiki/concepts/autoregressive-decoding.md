---
type: concept
title: Autoregressive Decoding
slug: autoregressive-decoding
date: 2026-04-20
updated: 2026-04-20
aliases: [AR decoding, auto-regressive decoding, 自回归解码]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Autoregressive Decoding** (自回归解码) — a generation procedure that produces output tokens sequentially, where each token is conditioned on the full prefix generated so far.

## Key Points

- The survey writes LLM inference as `y_t = argmax_y P(y | X_{t-1})` with `X_t = concat(X_{t-1}, y_t)`, making the sequential dependency explicit.
- It treats this one-token-at-a-time dependency as the core latency bottleneck that motivates speculative decoding, non-autoregressive methods, and early exiting.
- The paper emphasizes that autoregressive serving requires preserving and managing per-request state across iterations, especially the KV cache.
- Because output length is unknown in advance, autoregressive decoding complicates batching, scheduling, and latency prediction in production systems.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[miao-2023-efficient-2312-15234]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[miao-2023-efficient-2312-15234]].
