---
type: concept
title: Code Completion Benchmark
slug: code-completion-benchmark
date: 2026-04-20
updated: 2026-04-20
aliases: [code benchmark, 代码补全基准]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Code Completion Benchmark** (代码补全基准) — a standardized evaluation dataset and protocol for measuring how accurately models predict missing source code from a given programming context.

## Key Points

- CROSSCODEEVAL is positioned against single-file benchmarks such as HumanEval and MBPP, arguing that they miss repository-level dependencies.
- The benchmark contains around `10k` examples from `1,002` recent, permissively licensed repositories across four programming languages.
- It reports both code match metrics (`EM`, `ES`) and identifier match metrics (`EM`, `F1`), separating exact statement recovery from API-level correctness.
- The dataset is also designed to benchmark retrievers because performance depends strongly on whether relevant cross-file evidence is retrieved.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[ding-nd-crosscodeeval]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[ding-nd-crosscodeeval]].
