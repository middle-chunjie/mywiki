---
type: concept
title: Sequence Parallelism
slug: sequence-parallelism
date: 2026-04-20
updated: 2026-04-20
aliases: [sequence parallel training, 序列并行]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Sequence Parallelism** (序列并行) — a distributed training strategy that partitions long sequences across multiple devices so attention and activation memory can scale to much longer contexts.

## Key Points

- ProLong uses sequence parallelism only when moving to `512K` sequences, avoiding distributed attention overhead at `64K`.
- The implementation is based on DeepSpeed-Ulysses across groups of `8` GPUs on the same node.
- Retaining some `64K` examples in the `512K` stage reduces dependence on sequence parallelism and lowers communication overhead.
- The paper treats systems design as part of the training recipe, since extreme context lengths are otherwise infeasible to train.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[gao-2024-how-2410-02660]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[gao-2024-how-2410-02660]].
