---
type: concept
title: Rotary Positional Embedding
slug: rotary-positional-embedding
date: 2026-04-20
updated: 2026-04-20
aliases: [RoPE, rotary positional embedding, 旋转位置编码]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Rotary Positional Embedding** (旋转位置编码) — a positional encoding scheme that injects position information by rotating query and key representations in attention rather than adding learned or fixed absolute embeddings.

## Key Points

- OLMo replaces absolute positional embeddings with RoPE in both `1B` and `7B` decoder-only models.
- The paper treats RoPE as one of several mature LLM design choices inherited from recent families such as PaLM and LLaMA.
- RoPE is part of the architectural shift that distinguishes OLMo from the vanilla Transformer baseline it cites.
- The released `7B` model uses RoPE together with `32` attention heads and sequence length `2048`.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[groeneveld-2024-olmo-2402-00838]]
- [[an-2024-make-2404-16811]]
- [[unknown-nd-blockattention-2409-15355]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[groeneveld-2024-olmo-2402-00838]].
