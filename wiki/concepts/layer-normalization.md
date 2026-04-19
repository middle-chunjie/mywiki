---
type: concept
title: Layer Normalization
slug: layer-normalization
date: 2026-04-17
updated: 2026-04-17
aliases: [Layer Normalization, LayerNorm, 层归一化]
tags: [optimization, architecture]
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-17
---

## Definition

Layer Normalization (层归一化) — a normalization technique that rescales and re-centers activations across the feature dimension of each sample independently, yielding training signals that do not depend on batch statistics.

## Key Points

- Used in the [[transformer]] around every sub-layer: the sub-layer output becomes `LayerNorm(x + Sublayer(x))` (post-norm).
- Unlike batch normalization, it does not couple samples within a mini-batch, making it natural for variable-length sequence inputs.
- Paired with [[residual-connection]] to stabilize training of deep attention stacks.
- Introduced in Ba, Kiros & Hinton (2016); adopted as a core Transformer component.

## My Position

<!-- User's stance on this concept. Fed by personal writing; tag "(personal stance)" on such bullets. -->

## Contradictions

<!-- Disagreements among sources about this concept. -->

## Sources

- [[vaswani-2017-attention-1706-03762]]

## Evolution Log

- 2026-04-17 (1 source): established from [[vaswani-2017-attention-1706-03762]] as post-norm operator around every Transformer sub-layer.
