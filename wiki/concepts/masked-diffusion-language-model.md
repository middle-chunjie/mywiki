---
type: concept
title: Masked Diffusion Language Model
slug: masked-diffusion-language-model
date: 2026-04-20
updated: 2026-04-20
aliases: [masked diffusion language model, MDLM, 掩码扩散语言模型]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Masked Diffusion Language Model** (掩码扩散语言模型) — a discrete generative language model that starts from masked tokens and iteratively denoises them in parallel according to a learned reverse diffusion process.

## Key Points

- [[xiao-2026-embedding-2602-11047]] uses a masked diffusion LM as the core decoder for text embedding inversion instead of autoregressive generation plus correction.
- The forward corruption process masks each token with survival probability `alpha_t = exp(-5.0 t)`, and the reverse model predicts original tokens only at masked positions.
- Conditioning on the target embedding is injected into every transformer layer, so all positions can be refined jointly rather than left-to-right.
- The paper evaluates both pure Euler sampling and a remasking variant that reopens low-confidence positions for further denoising.
- The practical claim is that a masked diffusion LM reduces attack latency because it reconstructs a sequence in a small number of parallel denoising steps.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[xiao-2026-embedding-2602-11047]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[xiao-2026-embedding-2602-11047]].
