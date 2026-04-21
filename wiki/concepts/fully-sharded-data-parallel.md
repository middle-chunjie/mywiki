---
type: concept
title: Fully Sharded Data Parallel
slug: fully-sharded-data-parallel
date: 2026-04-20
updated: 2026-04-20
aliases: [FSDP, full sharding, 全分片数据并行]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Fully Sharded Data Parallel** (全分片数据并行) — a distributed training method that shards model parameters, gradients, and optimizer state across devices to reduce memory use during large-model training.

## Key Points

- OLMo trains with PyTorch FSDP together with the ZeRO strategy to shard weights and optimizer state across GPUs.
- At `7B`, this setup enables a micro-batch size of `4096` tokens per GPU on the reported hardware.
- The global batch for `1B` and `7B` training is approximately `4M` tokens.
- The paper emphasizes that the same FSDP-based codebase was validated on both AMD MI250X and NVIDIA A100 clusters.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[groeneveld-2024-olmo-2402-00838]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[groeneveld-2024-olmo-2402-00838]].
