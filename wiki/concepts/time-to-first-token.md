---
type: concept
title: Time-to-First-Token
slug: time-to-first-token
date: 2026-04-20
updated: 2026-04-20
aliases: [TTFT, time to first token]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Time-to-First-Token** — the latency between submitting an inference request and receiving the first generated token from the model system.

## Key Points

- TurboRAG treats TTFT as the main user-facing latency bottleneck in RAG because long retrieved contexts make naive prefill expensive.
- By reusing offline precomputed KV caches, the method cuts average LongBench TTFT from `1165 ms` to `134 ms`.
- The reported peak TTFT speedup is `9.4x` on MuSiQue, and the batch-scaling study still shows roughly `4x` speedups even when host-to-device KV transfer is included.
- The paper uses TTFT as the main systems metric linking reduced prefill compute to better responsiveness for latency-sensitive RAG applications.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[unknown-nd-turborag]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[unknown-nd-turborag]].
