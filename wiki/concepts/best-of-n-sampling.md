---
type: concept
title: Best-of-N Sampling
slug: best-of-n-sampling
date: 2026-04-20
updated: 2026-04-20
aliases: [best-of-n, BoN sampling, Best-of-N 采样]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Best-of-N Sampling** (Best-of-N 采样) — an inference strategy that samples `N` candidate outputs and returns the highest-scoring one according to a verifier or answer aggregator.

## Key Points

- The paper uses best-of-`N` as the main fixed-budget baseline for both verifier search and revision experiments.
- It replaces plain best-of-`N` with best-of-`N` weighted aggregation, summing verifier scores over candidates that share the same final answer.
- On easy questions and at higher search budgets, best-of-`N` is often more robust than beam-style PRM search.
- For revision models, pure parallel best-of-`N` underperforms prompt-adaptive mixtures of sequential and parallel generation.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[snell-2024-scaling-2408-03314]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[snell-2024-scaling-2408-03314]].
