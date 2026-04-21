---
type: concept
title: Adaptive Instance Normalization
slug: adaptive-instance-normalization
date: 2026-04-20
updated: 2026-04-20
aliases: [AdaIN, 自适应实例归一化]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Adaptive Instance Normalization** (自适应实例归一化) — a feature normalization operation that transfers channel-wise mean and variance statistics from a reference representation to a target representation.

## Key Points

- ReFIR uses AdaIN as its distribution-alignment module to match the fused cross-image attention output to the statistics of the original target-chain attention output.
- The motivation is that target and reference denoising chains have a domain gap caused by quality and content differences.
- By aligning mean and variance after fusion, the method reduces harmful distribution shift before replacing the original self-attention output.
- The paper treats AdaIN as a lightweight, training-free calibration step complementary to separate attention and spatial adaptive gating.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[guo-2024-refir-2410-05601]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[guo-2024-refir-2410-05601]].
