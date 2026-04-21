---
type: concept
title: Round-Trip Filtration
slug: round-trip-filtration
date: 2026-04-20
updated: 2026-04-20
aliases: [round trip filtration, cycle-consistent filtering, 往返过滤]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Round-Trip Filtration** (往返过滤) — a synthetic-data filtering procedure that keeps a generated example only if a second pass by the model reproduces the same answer from the generated context.

## Key Points

- The paper applies round-trip filtration after GPT-4 generates a synthetic context together with a question-answer pair.
- For each candidate example, GPT-4 is asked to answer the generated question again from the synthetic context without seeing the original answer.
- The example is retained only when the regenerated answer matches the original synthetic answer, i.e. `a_regen = a_syn`.
- The filtered variants are reported as `CC` setups in the experiments.
- This filter yields the best CovidQA result, but does not consistently help on PolicyQA or TechQA, indicating a precision-recall tradeoff.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[samuel-2023-can-2309-12426]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[samuel-2023-can-2309-12426]].
