---
type: concept
title: Code Efficiency
slug: code-efficiency
date: 2026-04-20
updated: 2026-04-20
aliases: [program efficiency]
tags: [code-generation, efficiency]
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Code efficiency** (代码效率) — the practical runtime and memory behavior of code under realistic workloads, beyond merely producing functionally correct outputs.

## Key Points

- [[huang-2024-effibench-2402-02037]] argues that correctness-only benchmarks can hide large efficiency differences between two valid programs.
- EffiBench operationalizes code efficiency with both execution-time metrics (`ET`, `NET`) and memory metrics (`MU`, `NMU`, `TMU`, `NTMU`).
- The benchmark shows that even strong models such as [[gpt-4-turbo]] still lag behind human canonical solutions on average and by large worst-case margins.
- Efficiency varies across algorithm families, indicating that code efficiency is not a single uniform capability.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[huang-2024-effibench-2402-02037]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[huang-2024-effibench-2402-02037]].
