---
type: concept
title: NTK-Aware RoPE Scaling
slug: ntk-aware-rope-scaling
date: 2026-04-20
updated: 2026-04-20
aliases: [NTK-Aware Scaled RoPE, NTK scaling, NTK-aware interpolation]
tags: [long-context, position-encoding, rope, context-length-extension]
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**NTK-Aware RoPE Scaling** (NTK感知的旋转位置编码缩放) — A fine-tuning-free method for extending LLM context windows that modifies the base of Rotary Position Embeddings (RoPE) using Neural Tangent Kernel (NTK) theory, aiming to distribute the resolution loss more evenly across frequency components than linear interpolation.

## Key Points

- Motivated by the observation that linear Position Interpolation (PI) compresses high-frequency components too aggressively, degrading position distinguishability at fine scales; NTK-aware scaling adjusts the RoPE base to spread information-theoretic loss more uniformly across all frequencies.
- Applied as a simple scaling of the base `θ` in RoPE: for target length `L'` vs. training length `L`, the base is set to `θ' = θ · (L'/L)^(d/(d-2))` where `d` is the head dimension.
- Requires no fine-tuning for a moderate extension ratio; can be combined with continued fine-tuning (YaRN) for stronger results.
- At extreme context lengths (100K+), both PI and NTK-aware scaling face OOM due to quadratic self-attention, unlike activation-compression-based methods.
- Empirically outperforms PI on perplexity across all tested lengths in the Activation Beacon paper: e.g., NTK achieves PG19 PPL=11.5 at 16K vs. PI's 19.5.

## My Position

<!-- User's stance on this concept. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[zhang-2024-long-2401-03462]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[zhang-2024-long-2401-03462]].
