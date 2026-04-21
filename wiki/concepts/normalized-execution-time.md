---
type: concept
title: Normalized Execution Time
slug: normalized-execution-time
date: 2026-04-20
updated: 2026-04-20
aliases: [NET]
tags: [evaluation, efficiency]
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Normalized Execution Time** (归一化执行时间) — the average ratio between generated-code execution time and canonical-solution execution time over evaluated samples.

## Key Points

- [[huang-2024-effibench-2402-02037]] defines `NET = (1/N) Σ (T_code / T_canonical)`.
- `NET > 1` means the generated code is slower than the canonical solution, while `NET < 1` means it is faster.
- EffiBench uses NET as a relative efficiency metric that is more comparable across problems than raw execution time alone.
- The paper reports [[gpt-4-turbo]] at `NET = 1.69` overall and `NET = 1.34` on the common correctly solved subset of closed-source models.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[huang-2024-effibench-2402-02037]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[huang-2024-effibench-2402-02037]].
