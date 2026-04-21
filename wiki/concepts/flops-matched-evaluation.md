---
type: concept
title: FLOPs-Matched Evaluation
slug: flops-matched-evaluation
date: 2026-04-20
updated: 2026-04-20
aliases: [compute-matched evaluation, FLOPs matching, FLOPs 匹配评测]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**FLOPs-Matched Evaluation** (FLOPs 匹配评测) — a comparison protocol that equalizes total floating-point compute so different improvement strategies can be evaluated under the same compute budget.

## Key Points

- The paper uses FLOPs matching to compare extra test-time compute against scaling model parameters.
- It models pretraining cost as `X = 6 N D_pretrain` and inference cost as `Y = 2 N D_inference`.
- For a parameter scaling factor `M`, the equivalent small-model inference multiplier is `M + 3 (D_pretrain / D_inference) (M - 1)`.
- The comparison shows that the better use of compute depends on both prompt difficulty and the inference-to-pretraining token ratio `R`.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[snell-2024-scaling-2408-03314]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[snell-2024-scaling-2408-03314]].
