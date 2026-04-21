---
type: concept
title: Arithmetic Intensity
slug: arithmetic-intensity
date: 2026-04-20
updated: 2026-04-20
aliases: [compute-to-memory ratio, 算术强度]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Arithmetic Intensity** (算术强度) — the ratio of computation to memory access that determines whether a workload is primarily limited by processor throughput or by memory bandwidth.

## Key Points

- [[yuan-2025-native-2502-11089]] uses arithmetic intensity as a first-class systems criterion when designing sparse attention for modern GPUs.
- The paper argues that training and prefilling attention are typically compute-bound, while autoregressive decoding is memory-bound because each new token still requires loading the full KV cache.
- NSA's hardware-aligned design tries to balance arithmetic intensity by reducing both redundant computation and scattered memory access.
- The reported decoding advantage is tied directly to reduced KV-cache traffic, while training and prefilling gains come from kernels that better utilize Tensor Cores.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[yuan-2025-native-2502-11089]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[yuan-2025-native-2502-11089]].
