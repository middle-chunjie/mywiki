---
type: concept
title: Latent Attention
slug: latent-attention
date: 2026-04-20
updated: 2026-04-20
aliases: [latent attention]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Latent Attention** — a pooling mechanism that lets token representations attend to a trainable latent array before sequence-level aggregation, producing a more expressive embedding than plain mean or EOS pooling.

## Key Points

- [[lee-2024-nvembed-2405-17428]] places a latent-attention layer on top of a decoder-only LLM to replace simple sequence pooling for embedding extraction.
- The layer uses decoder hidden states as queries and a trainable dictionary as keys and values, with `` `O = softmax(QK^T)V` `` before an MLP and final mean pooling.
- In NV-Embed, the latent array uses `512` latents, latent width `4096`, and `8` attention heads.
- The paper argues latent attention reduces information dilution from mean pooling and recency bias from EOS-only pooling.
- In the v2 ablation, latent attention improves retrieval over mean pooling (`62.65` vs. `61.82`) while extra self-attention does not yield similar gains.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[lee-2024-nvembed-2405-17428]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[lee-2024-nvembed-2405-17428]].
