---
type: concept
title: Multi-Head Latent Attention
slug: multi-head-latent-attention
date: 2026-04-20
updated: 2026-04-20
aliases: [MLA, multi head latent attention]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Multi-Head Latent Attention** — an attention mechanism that compresses keys and values into a shared low-rank latent representation while preserving positional information through a decoupled RoPE path.

## Key Points

- MLA replaces standard MHA in DeepSeek-V2 to shrink inference-time KV cache without losing the quality associated with full multi-head attention.
- The method computes compressed latent states with `c_t^KV = W^DKV h_t` and reconstructs content keys and values from that latent vector rather than caching full per-head KV tensors.
- DeepSeek-V2 sets `d_c = 512`, `d'_c = 1536`, and `d_h^R = 64`, so the cache scales as `(d_c + d_h^R) l` rather than `2 n_h d_h l`.
- Decoupled RoPE lets the model retain rotary positional information without breaking the absorbable projection structure that makes the cache compression useful at inference time.
- The paper reports MLA as both more efficient than MHA and stronger than MHA, GQA, and MQA under the authors' ablations.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[deepseek-ai-2024-deepseekv-2405-04434]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[deepseek-ai-2024-deepseekv-2405-04434]].
