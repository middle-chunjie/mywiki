---
type: concept
title: Canonical Solution
slug: canonical-solution
date: 2026-04-20
updated: 2026-04-20
aliases: [reference solution]
tags: [benchmark, evaluation]
source_count: 1
confidence: low
domain_volatility: low
last_reviewed: 2026-04-20
---

## Definition

**Canonical solution** (规范解) — an executable reference implementation used as the baseline for comparing generated code on normalized evaluation metrics.

## Key Points

- In [[huang-2024-effibench-2402-02037]], each problem is paired with a human-written canonical solution collected from top-starred LeetCode discussion answers.
- The authors manually repair missing imports and platform-specific dependencies so the canonical solutions run outside LeetCode.
- EffiBench normalizes execution time and memory usage against the canonical solution to obtain `NET`, `NMU`, and `NTMU`.
- The paper treats the canonical solution as a practical baseline, while acknowledging that it may not be the globally optimal implementation.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[huang-2024-effibench-2402-02037]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[huang-2024-effibench-2402-02037]].
