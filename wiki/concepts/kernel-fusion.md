---
type: concept
title: Kernel Fusion
slug: kernel-fusion
date: 2026-04-20
updated: 2026-04-20
aliases: [fused kernels, 内核融合]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Kernel Fusion** (内核融合) — an implementation strategy that merges multiple operators into fewer GPU kernels to reduce launch overhead and memory traffic.

## Key Points

- The survey identifies inference as especially fusion-friendly because backward computation is absent and more operator boundaries can be collapsed safely.
- It describes fusion of same-shape GEMMs and bias/add/norm/activation patterns as a standard route to lower launch overhead and HBM traffic.
- The paper emphasizes fused attention kernels as one of the most important practical optimizations in modern LLM serving engines.
- Representative systems such as FasterTransformer, TenTrans, TurboTransformers, LightSeq, and compiler stacks like Welder are cited as concrete examples of this design pattern.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[miao-2023-efficient-2312-15234]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[miao-2023-efficient-2312-15234]].
