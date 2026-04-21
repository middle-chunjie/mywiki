---
type: concept
title: Code Refinement
slug: code-refinement
date: 2026-04-20
updated: 2026-04-20
aliases: [program refinement, repair generation, 代码精炼]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Code Refinement** (代码精炼) — the task of revising an incorrect program into a corrected one using contextual feedback such as failing tests, error messages, or natural-language diagnosis.

## Key Points

- LeDex frames code refinement as a supervised and reinforcement-learned behavior rather than a prompting-only capability.
- The training pipeline keeps only refinements that pass all benchmark unit tests, so verified repair success defines positive supervision.
- The paper uses two instruction formats: direct refinement and explanation followed by refinement, with both contributing to robustness.
- Refinement quality is rewarded by combining CodeBLEU similarity to verified repairs with unit-test pass rate via `R(r) = 5 · (S_cb + S_ut) - 5`.
- The learned refinement policy improves both one-step pass@k and multi-round iterative debugging performance on MBPP, HumanEval, MBPP+, and HumanEval+.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[jiang-2024-training-2405-18649]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[jiang-2024-training-2405-18649]].
