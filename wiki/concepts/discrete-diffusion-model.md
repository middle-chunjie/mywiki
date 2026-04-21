---
type: concept
title: Discrete Diffusion Model
slug: discrete-diffusion-model
date: 2026-04-20
updated: 2026-04-20
aliases: [discrete diffusion model, 离散扩散模型]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Discrete Diffusion Model** (离散扩散模型) — a diffusion model defined over categorical states such as tokens, where corruption and denoising operate on discrete symbols rather than continuous noise vectors.

## Key Points

- [[xiao-2026-embedding-2602-11047]] adapts discrete diffusion to text inversion by masking tokens instead of adding Gaussian noise to embeddings or hidden states.
- The forward process uses an absorbing `[MASK]` state and a log-linear survival schedule, making the corruption distribution easy to sample during training.
- The reverse model predicts token distributions in parallel over the full vocabulary, which matches the discrete reconstruction target of exact text recovery.
- The paper argues that discrete diffusion avoids the left-to-right error propagation of autoregressive inversion baselines while preserving iterative correction.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[xiao-2026-embedding-2602-11047]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[xiao-2026-embedding-2602-11047]].
