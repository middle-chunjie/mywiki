---
type: concept
title: Test-Case Generation
slug: test-case-generation
date: 2026-04-20
updated: 2026-04-20
aliases: [unit test generation, test case generation, 测试用例生成]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Test-Case Generation** (测试用例生成) — the automatic production of executable test inputs and assertions that probe whether a program satisfies its intended behavior.

## Key Points

- [[ficek-2025-scoring-2502-13820]] evaluates test-case generation by how well generated tests score and rank candidate solutions, not just by coverage or validity in isolation.
- The paper uses a fixed prompt with two HumanEval-style examples and typically generates `10` tests per problem at temperature `1.0`.
- Generated tests are executed against benchmark solutions with a `3`-second timeout to derive verifier scores for Top-1, Bottom-1, Spearman, and MAE evaluation.
- Reasoning models such as DeepSeek-R1 and o3-mini outperform standard instruction-tuned models, showing that stronger reasoning improves the discriminative power of generated test suites.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[ficek-2025-scoring-2502-13820]]
- [[m-ndler-2024-code-2406-12952]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[ficek-2025-scoring-2502-13820]].
