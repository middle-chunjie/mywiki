---
type: concept
title: Inference Latency
slug: inference-latency
date: 2026-04-20
updated: 2026-04-20
aliases: []
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Inference Latency** (推理时延) — the end-to-end time between submitting a model request and receiving its generated response.

## Key Points

- The paper identifies sequential decoding as a dominant contributor to latency for long assistant answers.
- SoT targets latency without modifying model weights, attention kernels, or serving hardware.
- The motivating examples reduce latency from `22 s` to `12 s` on Claude and from `43 s` to `16 s` on Vicuna-33B V1.3.
- Profiling on A100 shows decoding latency dominates prefilling latency because decoding repeatedly reloads model weights.
- The paper treats latency, rather than throughput, as the primary optimization objective.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[ning-2024-skeletonofthought-2307-15337]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[ning-2024-skeletonofthought-2307-15337]].
