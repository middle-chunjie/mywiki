---
type: concept
title: Decoder-Only Transformer
slug: decoder-only-transformer
date: 2026-04-20
updated: 2026-04-20
aliases: [causal transformer, decoder-only LM, 仅解码器 Transformer]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Decoder-Only Transformer** (仅解码器 Transformer) — an autoregressive Transformer architecture that predicts the next token using only causal self-attention blocks, without a separate encoder stack.

## Key Points

- OLMo adopts a decoder-only architecture derived from the Transformer and scales it to `1B` and `7B` parameters.
- The `1B` model uses `16` layers, hidden size `2048`, and `16` attention heads, while the `7B` model uses `32` layers, hidden size `4096`, and `32` heads.
- Sequence length is `2048`, and the model keeps a sequential decoder block design rather than a parallel block layout.
- OLMo modernizes the basic decoder with no biases, non-parametric layer normalization, SwiGLU activations, and RoPE.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[groeneveld-2024-olmo-2402-00838]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[groeneveld-2024-olmo-2402-00838]].
