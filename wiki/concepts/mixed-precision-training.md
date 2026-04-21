---
type: concept
title: Mixed-Precision Training
slug: mixed-precision-training
date: 2026-04-20
updated: 2026-04-20
aliases: [mixed precision, BF16 training, 混合精度训练]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Mixed-Precision Training** (混合精度训练) — a training regime that uses lower-precision arithmetic for most tensor operations while retaining full precision for sensitive computations to improve throughput without destabilizing optimization.

## Key Points

- OLMo uses BF16 for most forward and backward computations through FSDP and PyTorch AMP.
- Softmax is forced to run in full precision to improve numerical stability.
- Sharded optimizer state and local full-sized parameters are kept in full precision, then cast to BF16 when transformer block weights are materialized.
- Gradient reduction across GPUs is also performed in full precision.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[groeneveld-2024-olmo-2402-00838]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[groeneveld-2024-olmo-2402-00838]].
