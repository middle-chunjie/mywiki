---
type: concept
title: Unit-Test-Based Evaluation
slug: unit-test-based-evaluation
date: 2026-04-20
updated: 2026-04-20
aliases: [test-based attribution, 单元测试评估]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Unit-Test-Based Evaluation** (单元测试评估) — an evaluation protocol that judges generated code by whether it passes a designated suite of executable tests.

## Key Points

- [[hooda-2024-do-2402-05980]] uses unit-test correctness as the attribution function for both original and counterfactual completions.
- The paper argues this is more faithful than string-match metrics such as CodeBLEU or chrF because semantically equivalent programs may differ syntactically.
- Only problems where the model succeeds on the original input, the perturbed input, or both are kept for AME computation.
- The framework therefore depends on benchmarks that provide runnable tests for correctness checking.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[hooda-2024-do-2402-05980]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[hooda-2024-do-2402-05980]].
