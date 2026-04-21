---
type: concept
title: Test-Time Augmentation
slug: test-time-augmentation
date: 2026-04-20
updated: 2026-04-20
aliases: [TTA, 测试时增强]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Test-Time Augmentation** (测试时增强) — a prediction-time strategy that evaluates multiple transformed views of the same input and aggregates their outputs to improve robustness.

## Key Points

- The paper applies the same semantic-preserving code transformations used in training to the test phase.
- For each test example, it samples `3` augmented copies and combines their predictions with the original input's prediction.
- In code search, the model averages query-code similarity across transformed code variants before ranking candidates.
- Ablation results show TTA is the largest single contributor on POJ-104, where removing it lowers MAP from `92.91` to `87.21`.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[wang-2022-bridging-2112-02268]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[wang-2022-bridging-2112-02268]].
