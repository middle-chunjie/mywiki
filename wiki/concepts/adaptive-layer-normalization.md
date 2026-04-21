---
type: concept
title: Adaptive Layer Normalization
slug: adaptive-layer-normalization
date: 2026-04-20
updated: 2026-04-20
aliases: [AdaLN, adaptive layer normalization, 自适应层归一化]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Adaptive Layer Normalization** (自适应层归一化) — a conditioning mechanism that modulates layer-normalized hidden states with scale and shift parameters derived from external control signals.

## Key Points

- [[xiao-2026-embedding-2602-11047]] uses AdaLN to inject both timestep information and the target embedding into each transformer layer of the inversion decoder.
- For each layer, the model computes `gamma_t`, `beta_t`, `gamma_c`, and `beta_c`, then combines them as `gamma = gamma_t + gamma_c` and `beta = beta_t + beta_c`.
- The conditioning vector is produced by projecting the target embedding into hidden size `768` with a two-layer MLP before modulation.
- AdaLN is the main reason the paper can treat the decoder as encoder-agnostic at inference time: the target embedding enters as a generic conditioning vector rather than through cross-attention to encoder states.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[xiao-2026-embedding-2602-11047]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[xiao-2026-embedding-2602-11047]].
