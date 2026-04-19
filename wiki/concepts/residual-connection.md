---
type: concept
title: Residual Connection
slug: residual-connection
date: 2026-04-17
updated: 2026-04-17
aliases: [Residual Connection, Skip Connection, 残差连接, 跳跃连接]
tags: [optimization, architecture]
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-17
---

## Definition

Residual Connection (残差连接) — an additive shortcut that adds a sub-layer's input to its output, producing `x + Sublayer(x)`, to ease gradient flow and enable training of deeper networks.

## Key Points

- Originates in deep residual networks for image recognition; the [[transformer]] adopts it around every encoder and decoder sub-layer.
- In [[vaswani-2017-attention-1706-03762]], each sub-layer output is `LayerNorm(x + Sublayer(x))` — a post-norm arrangement.
- Requires matching dimensionality between input and sub-layer output; this is why all sub-layers and embeddings in the Transformer produce `d_model`-dimensional outputs.
- Supports gradient flow through deep stacks (e.g., `N = 6` layers in base, 12+ in later LLMs).

## My Position

<!-- User's stance on this concept. Fed by personal writing; tag "(personal stance)" on such bullets. -->

## Contradictions

<!-- Disagreements among sources about this concept. -->

## Sources

- [[vaswani-2017-attention-1706-03762]]

## Evolution Log

- 2026-04-17 (1 source): established from [[vaswani-2017-attention-1706-03762]] as wrapper around every sub-layer in post-norm form.
