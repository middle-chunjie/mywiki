---
type: concept
title: Memory Region Prediction
slug: memory-region-prediction
date: 2026-04-20
updated: 2026-04-20
aliases: [memory region prediction, 内存区域预测]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Memory Region Prediction** (内存区域预测) — the instruction-level task of predicting which memory region type a memory-accessing instruction refers to, such as stack, heap, global, or other.

## Key Points

- The paper treats memory-region prediction as a token-level invariant analysis task over binary code.
- SymC uses a per-token predictor `F:R^d -> L` on top of equivariant attention outputs rather than pooled sequence embeddings.
- On unseen permutations, SymC remains fixed at `0.86` F1 across all permutation ratios.
- The efficiency study uses this task to compare training cost, showing SymC reaches `0.5` F1 with `0.07` GPU hours versus `89.67` for PalmTree-O.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[pei-2024-exploiting-2308-03312]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[pei-2024-exploiting-2308-03312]].
