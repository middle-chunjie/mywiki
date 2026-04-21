---
type: concept
title: Cross-Attention
slug: cross-attention
date: 2026-04-20
updated: 2026-04-20
aliases: [交叉注意力, cross attention]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Cross-Attention** (交叉注意力) — an attention mechanism in which queries from one representation attend to keys and values from another representation.

## Key Points

- ReFIR introduces inter-chain cross-attention from the target denoising chain to the retrieved-reference chain so the restoration model can use external textures.
- The paper avoids naïvely concatenating target and source keys because the target queries exhibit domain preference for same-chain features.
- Its separate-attention design keeps intra-chain self-attention and inter-chain cross-attention as distinct paths before gated fusion.
- Ablations show decoder-side cross-image attention is beneficial, while encoder-side injection can harm layout fidelity.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[guo-2024-refir-2410-05601]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[guo-2024-refir-2410-05601]].
