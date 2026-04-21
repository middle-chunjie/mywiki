---
type: concept
title: Paged Optimizer
slug: paged-optimizer
date: 2026-04-20
updated: 2026-04-20
aliases: [paged optimizers, 分页优化器]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Paged Optimizer** (分页优化器) — an optimizer implementation that pages optimizer states between GPU and host memory to handle transient memory spikes without failing training.

## Key Points

- QLoRA uses paged optimizers to manage the activation-gradient spikes that arise with long sequences and gradient checkpointing.
- The paper implements paging via NVIDIA unified memory so optimizer states can be evicted to CPU RAM and fetched back when needed.
- This mechanism is presented as necessary for fitting the largest QLoRA runs, especially `33B` and `65B`, onto commodity or single-GPU hardware.
- The authors note that runtime overhead is small in their 65B analysis at batch size `16`, but leave a fuller characterization for future work.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[dettmers-2023-qlora-2305-14314]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[dettmers-2023-qlora-2305-14314]].
