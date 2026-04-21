---
type: concept
title: Data Leakage
slug: data-leakage
date: 2026-04-20
updated: 2026-04-20
aliases: [data contamination, 数据泄漏]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Data Leakage** (数据泄漏) — a benchmark-evaluation failure mode in which test items or close variants appear in model training data, causing inflated performance estimates.

## Key Points

- The paper treats leakage as a central threat to trustworthy evaluation of code LLMs because modern training corpora absorb large amounts of open-source code.
- EvoCodeBench mitigates leakage by sampling from recent repositories created between `October 2023` and `March 2024`.
- Using the CDD detector, the paper reports leakage rates below `3%` on EvoCodeBench-2403, compared with `41.47%` on HumanEval for `gpt-3.5`.
- The authors argue zero leakage is unrealistic because repositories can still contain highly similar programs such as common utility functions.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[li-2024-evocodebench-2410-22821]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[li-2024-evocodebench-2410-22821]].
