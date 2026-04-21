---
type: concept
title: Benchmark Evaluation
slug: benchmark-evaluation
date: 2026-04-20
updated: 2026-04-20
aliases: [automatic evaluation, benchmarking, 基准评测]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Benchmark Evaluation** (基准评测) — measuring model performance on standardized held-out datasets with fixed tasks and scoring metrics.

## Key Points

- The paper uses `5-shot` MMLU, `3-shot` Natural Questions, and `0-shot` HumanEval to probe broad knowledge, factual QA, and coding.
- These automatic evaluations reveal little or no broad capability transfer from ChatGPT imitation, contradicting the optimistic human preference results.
- Benchmark scores are more sensitive than pairwise preference judgments to regressions caused by conversational imitation data.
- The study uses benchmark evaluation to argue that style parity should not be confused with capability parity.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[gudibande-2023-false-2305-15717]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[gudibande-2023-false-2305-15717]].
