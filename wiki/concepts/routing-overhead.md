---
type: concept
title: Routing Overhead
slug: routing-overhead
date: 2026-04-20
updated: 2026-04-20
aliases: [router overhead, 路由开销]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Routing Overhead** (路由开销) — the additional compute, communication, and memory cost introduced by deciding which experts process each token in a sparse MoE model.

## Key Points

- The paper argues that routing overhead is the main practical limit on arbitrarily increasing MoE granularity.
- Its FLOP model adds a routing term proportional to `d_model * E * G * c_r`, with `c_r = 14`.
- Because routing cost scales with `G`, wall-clock efficiency can worsen even when higher granularity improves loss at fixed training steps.
- This overhead explains why compute-optimal granularity is finite and why extreme `G` can degrade performance.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[krajewski-2024-scaling-2402-07871]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[krajewski-2024-scaling-2402-07871]].
