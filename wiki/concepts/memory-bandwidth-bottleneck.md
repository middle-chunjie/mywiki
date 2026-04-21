---
type: concept
title: Memory Bandwidth Bottleneck
slug: memory-bandwidth-bottleneck
date: 2026-04-20
updated: 2026-04-20
aliases: [内存带宽瓶颈]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Memory Bandwidth Bottleneck** (内存带宽瓶颈) — a systems regime where inference latency is limited more by moving parameters and cache state than by the availability of arithmetic operations.

## Key Points

- The paper argues large Transformer decoding is often bottlenecked on memory bandwidth and communication rather than pure compute.
- This observation motivates speculative decoding: spare arithmetic can be spent on parallel verification while reducing serial target passes.
- Even when arithmetic work increases, the number of target-model weight and KV-cache reads per emitted token can decrease.
- The method is therefore most attractive when additional compute resources are available but memory movement dominates latency.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[leviathan-2023-fast-2211-17192]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[leviathan-2023-fast-2211-17192]].
