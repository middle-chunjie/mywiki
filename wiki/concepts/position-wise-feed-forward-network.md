---
type: concept
title: Position-wise Feed-Forward Network
slug: position-wise-feed-forward-network
date: 2026-04-17
updated: 2026-04-17
aliases: [Position-wise Feed-Forward Network, FFN, Position-wise FFN, 位置前馈网络, 前馈网络]
tags: [architecture, nlp]
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-17
---

## Definition

Position-wise Feed-Forward Network (位置前馈网络) — a two-layer fully connected network with a ReLU activation applied independently and identically to each position in a sequence, functioning as a per-position non-linear transformation inside each Transformer block.

## Key Points

- Formula: `FFN(x) = max(0, xW_1 + b_1) W_2 + b_2`.
- Equivalent to two kernel-size-1 convolutions along the sequence axis.
- Base Transformer sizes: input/output dimension `d_model = 512`, inner dimension `d_ff = 2048` (≈ 4× widening).
- Parameters are shared across positions within a layer but differ across layers.
- Paired with [[multi-head-attention]] in each Transformer sub-block; the FFN is where most per-position non-linearity and parameters live.

## My Position

<!-- User's stance on this concept. Fed by personal writing; tag "(personal stance)" on such bullets. -->

## Contradictions

<!-- Disagreements among sources about this concept. -->

## Sources

- [[vaswani-2017-attention-1706-03762]]

## Evolution Log

- 2026-04-17 (1 source): established from [[vaswani-2017-attention-1706-03762]] with `d_model=512`, `d_ff=2048`, ReLU.
