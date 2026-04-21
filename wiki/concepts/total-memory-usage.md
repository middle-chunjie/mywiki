---
type: concept
title: Total Memory Usage
slug: total-memory-usage
date: 2026-04-20
updated: 2026-04-20
aliases: [TMU]
tags: [evaluation, memory]
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Total memory usage** (总内存使用量) — the time-integrated memory consumption of a program over its whole execution, capturing both how much memory is used and for how long.

## Key Points

- [[huang-2024-effibench-2402-02037]] defines `TMU = (1/N) Σ ∫_0^{T_total} M(t) dt`.
- TMU differs from max-memory metrics because it accounts for duration, not only peak magnitude.
- EffiBench also reports normalized total memory usage (`NTMU`) by comparing generated code with the canonical solution.
- The benchmark shows large TMU gaps for some models; for example, [[gpt-4-turbo]] still has `NTMU = 3.18` on average.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[huang-2024-effibench-2402-02037]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[huang-2024-effibench-2402-02037]].
