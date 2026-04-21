---
type: concept
title: RoPE
slug: rope
date: 2026-04-20
updated: 2026-04-20
aliases: [Rotary Position Embedding, rotary positional embedding, 旋转位置编码]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**RoPE** (旋转位置编码) — a positional encoding scheme that injects token position through learned rotations in attention space, enabling relative position sensitivity in transformer-style models.

## Key Points

- This paper modifies the RoPE frequency base as the main mechanism for extending Llama-3 from an `8K` context window to `64K` and then `512K`.
- The final recipe uses a base of `8 x 10^6` at `64K` and `1.28 x 10^8` at `512K`, far above the original Llama setting.
- RoPE scaling is not enough by itself; the paper combines it with billions of continued-training tokens and downstream evaluation after SFT.
- Ablations in the appendix show that poor RoPE base choices sharply degrade long-context quality even when other training components remain fixed.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[gao-2024-how-2410-02660]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[gao-2024-how-2410-02660]].
