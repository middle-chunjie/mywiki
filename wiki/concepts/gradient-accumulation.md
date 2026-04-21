---
type: concept
title: Gradient Accumulation
slug: gradient-accumulation
date: 2026-04-20
updated: 2026-04-20
aliases: [梯度累积]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Gradient Accumulation** (梯度累积) — a training procedure that sums parameter gradients over several small forward-backward passes before one optimizer update to simulate a larger effective batch size.

## Key Points

- The paper uses gradient accumulation as a baseline for memory-limited contrastive learning.
- It argues that accumulation is not equivalent to true large-batch contrastive training because each micro-batch contains fewer in-batch negatives.
- In the retrieval experiment, the accumulation baseline improves over purely sequential updates but still trails gradient cache on Top-5/20/100 retrieval accuracy.
- The method remains useful for parameter-memory constraints, but this paper highlights its mismatch with losses whose statistics depend on the full batch composition.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[gao-2021-scaling-2101-06983]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[gao-2021-scaling-2101-06983]].
