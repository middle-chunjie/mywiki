---
type: concept
title: Data Decontamination
slug: data-decontamination
date: 2026-04-20
updated: 2026-04-20
aliases: [benchmark decontamination, 数据去污染]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Data Decontamination** (数据去污染) — the removal of benchmark or test-set overlap from training data to reduce leakage and make evaluation reflect generalization more faithfully.

## Key Points

- The phi-4 team explicitly improves decontamination relative to earlier Phi models and treats contamination risk as a major threat to benchmark credibility.
- The report lists a broad set of protected benchmarks, including MMLU, MMLU-Pro, GSM8K, GPQA, HumanEval, MBPP, MATH, ARC, and ArenaHard.
- Fresh AMC-10/12 2024 evaluations are used as an additional contamination-resistant sanity check because those tests postdate the training corpus.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[abdin-2024-phi-2412-08905]]
- [[groeneveld-2024-olmo-2402-00838]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[abdin-2024-phi-2412-08905]].
