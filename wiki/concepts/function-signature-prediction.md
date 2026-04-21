---
type: concept
title: Function Signature Prediction
slug: function-signature-prediction
date: 2026-04-20
updated: 2026-04-20
aliases: [function signature prediction, 函数签名预测]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Function Signature Prediction** (函数签名预测) — the task of inferring a function's argument count and source-level types from its code representation.

## Key Points

- The paper uses function signature prediction as one of its core binary-analysis benchmarks.
- SymC applies a mean-pooled invariant predictor `F:R^d -> L` on top of equivariant representations for this task.
- On unseen compilers, the paper reports gains of up to `38.6%` over PalmTree.
- On unseen permutations, SymC remains stable at `0.88` F1 across all tested permutation ratios.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[pei-2024-exploiting-2308-03312]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[pei-2024-exploiting-2308-03312]].
