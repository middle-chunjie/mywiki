---
type: concept
title: All-Gather
slug: all-gather
date: 2026-04-20
updated: 2026-04-20
aliases: [全收集, allgather]
tags: []
source_count: 1
confidence: low
domain_volatility: low
last_reviewed: 2026-04-20
---

## Definition

**All-Gather** (全收集) — a distributed communication primitive that collects tensors from every device and makes the concatenated result available on each participating device.

## Key Points

- In the paper's multi-GPU version, all-gather is applied after graph-less forward so every GPU can compute contrastive loss over the global batch representations.
- Each device then caches gradients only for its local representations even though the loss is defined over `F_all` and `G_all`.
- No extra communication is needed during sub-batch replay; standard distributed gradient reduction happens only at optimizer time.
- This communication pattern preserves exact large-batch contrastive gradients across devices while keeping encoder backpropagation local.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[gao-2021-scaling-2101-06983]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[gao-2021-scaling-2101-06983]].
