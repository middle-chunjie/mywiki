---
type: concept
title: Execution-Time Profiling
slug: execution-time-profiling
date: 2026-04-20
updated: 2026-04-20
aliases: [runtime profiling, 执行时间剖析]
tags: [profiling, runtime, code]
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Execution-Time Profiling** (执行时间剖析) — the measurement of per-line or per-operation runtime cost during program execution for identifying performance bottlenecks.

## Key Points

- [[huang-2024-effilearner-2405-15189]] uses Python `line_profiler` to collect line numbers, execution counts, and total time consumed.
- The profiling runs over the benchmark's open test cases rather than a single example.
- The resulting trace lets the LLM map high runtime cost back to specific loops, operators, or control-flow choices.
- Time-only feedback helps, but the paper reports the best results when runtime profiling is combined with memory profiling inside EFFI-LEARNER.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[huang-2024-effilearner-2405-15189]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[huang-2024-effilearner-2405-15189]].
