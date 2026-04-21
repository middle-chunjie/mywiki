---
type: concept
title: Code Generation Benchmark
slug: code-generation-benchmark
date: 2026-04-20
updated: 2026-04-20
aliases: [coding benchmark, 代码生成基准]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Code Generation Benchmark** (代码生成基准) — an evaluation dataset and protocol used to measure how well systems generate executable code under controlled tasks, contexts, and metrics.

## Key Points

- EvoCodeBench targets repo-level code generation rather than standalone snippets, so benchmark items include repository context and dependency information.
- The first release contains `275` tasks from `25` recent Python repositories and is designed to approximate real-world code and dependency distributions.
- The benchmark emphasizes trustworthy evaluation by reducing training-data leakage through continuously refreshed repository sources.
- It expands the usual benchmark design by attaching domain labels and reporting domain-specific analyses rather than only global averages.
- Each sample includes executable tests, enabling direct functional evaluation through `Pass@k`.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[li-2024-evocodebench-2410-22821]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[li-2024-evocodebench-2410-22821]].
