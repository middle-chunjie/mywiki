---
type: concept
title: Scaled Dot-Product Attention
slug: scaled-dot-product-attention
date: 2026-04-17
updated: 2026-04-17
aliases: [Scaled Dot-Product Attention, 缩放点积注意力]
tags: [attention, nlp]
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-17
---

## Definition

Scaled Dot-Product Attention (缩放点积注意力) — an attention function computing `softmax(QKᵀ / √d_k) V`, where dot products between queries and keys are rescaled by `1/√d_k` before softmax to stabilize gradients for large `d_k`.

## Key Points

- Equivalent to standard dot-product (multiplicative) attention except for the `1/√d_k` scaling factor.
- Faster and more memory-efficient than additive attention in practice, since it reduces to dense matrix multiplication.
- Without scaling, large `d_k` drives dot products to large magnitudes, pushing softmax into low-gradient regions.
- For small `d_k`, additive and dot-product attention perform similarly; scaling is the fix that makes dot-product viable at higher `d_k`.
- The building block inside every [[multi-head-attention]] head in the [[transformer]].

## My Position

<!-- User's stance on this concept. Fed by personal writing; tag "(personal stance)" on such bullets. -->

## Contradictions

<!-- Disagreements among sources about this concept. -->

## Sources

- [[vaswani-2017-attention-1706-03762]]

## Evolution Log

- 2026-04-17 (1 source): established from [[vaswani-2017-attention-1706-03762]] with `1/√d_k` rescaling as gradient-stabilization fix.
