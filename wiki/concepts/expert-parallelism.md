---
type: concept
title: Expert Parallelism
slug: expert-parallelism
date: 2026-04-20
updated: 2026-04-20
aliases: [expert parallelism, 专家并行]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Expert Parallelism** (专家并行) — a distributed training strategy for MoE models in which different experts are placed on different devices and tokens are communicated to the devices hosting their selected experts.

## Key Points

- DeepSeek-V2 uses `8`-way expert parallelism so that routed experts in each MoE layer are spread across `8` devices.
- Because fine-grained routing can increase communication overhead, the paper adds device-limited routing so each token is sent to at most `3` devices.
- The training objective includes device-level and communication-balance auxiliary losses to keep expert-parallel execution efficient.
- Shared-expert computation is overlapped with all-to-all communication to improve end-to-end throughput in the HAI-LLM training stack.
- Expert parallelism is central to making a `236B`-parameter MoE model train economically while activating only `21B` parameters per token.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[deepseek-ai-2024-deepseekv-2405-04434]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[deepseek-ai-2024-deepseekv-2405-04434]].
