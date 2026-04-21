---
type: concept
title: Speculation Stride Scheduling
slug: speculation-stride-scheduling
date: 2026-04-20
updated: 2026-04-20
aliases: [OS3, Optimal Speculation Stride Scheduler, 投机步长调度]
tags: [serving, speculation, scheduling, optimization]
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Speculation Stride Scheduling** (投机步长调度) — an adaptive online algorithm that sets the number of consecutive speculative steps (stride `s`) between verification rounds to maximize verified documents per unit time, given estimated speculation accuracy and step latencies.

## Key Points

- The speculation stride `s` controls the tradeoff between overhead and saving: large `s` risks high rollback cost if mismatch occurs early; small `s` under-exploits batch efficiency.
- For synchronous verification, the objective is `(1 − γ^s) / [(1 − γ)(sa + b)]`; for asynchronous, the denominator becomes the expected latency `γ^s·((s−1)a + max(a,b)) + (1−γ^s)·(sa+b)`, where `a` = speculation step latency, `b` = verification latency, `γ` = per-step speculation accuracy.
- OS3 estimates `γ` via a sliding window of the last `w = 5` verification results, capped at `γ_max = 0.6`; `a` and `b` are measured wall-clock averages.
- In ablations, OS3 provides the largest individual speedup among the three RaLMSpec components, particularly for approximate and sparse retrievers where a fixed stride of 3 would be harmful.
- Without OS3, RaLMSpec with approximate dense retriever falls below baseline (e.g., `0.61×`); OS3 restores it to ≥ `1.0×` and achieves up to `1.39×`.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[zhang-2024-accelerating-2401-14021]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[zhang-2024-accelerating-2401-14021]].
