---
type: concept
title: Overhead Profiling
slug: overhead-profiling
date: 2026-04-20
updated: 2026-04-20
aliases: [runtime overhead profiling, 开销剖析]
tags: [profiling, efficiency, code]
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Overhead Profiling** (开销剖析) — the measurement of execution-time and memory-use overhead during program execution in order to localize inefficiencies in concrete code regions.

## Key Points

- [[huang-2024-effilearner-2405-15189]] combines runtime and memory profiles instead of giving the model only aggregate metrics such as ET or MU.
- The collected profiles are line-level and aggregated across all open test cases for a task.
- These profiles are injected directly into the refinement prompt as actionable evidence about bottlenecks.
- The paper shows profiler-rich feedback outperforms unsupervised self-refinement and result-aware feedback that only includes scalar efficiency numbers.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[huang-2024-effilearner-2405-15189]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[huang-2024-effilearner-2405-15189]].
