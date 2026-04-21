---
type: concept
title: Pre-Norm
slug: pre-norm
date: 2026-04-20
updated: 2026-04-20
aliases: [Pre-Layer Normalization, Pre-LN, 前归一化]
tags: [neural-architecture, normalization, transformer]
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Pre-Norm** (前归一化) — a residual block arrangement that applies layer normalization to the input before passing it to the sub-layer, producing `x + Sublayer(LayerNorm(x))`, as opposed to the original Post-Norm which normalizes after the residual addition.

## Key Points

- Pre-Norm mitigates gradient vanishing in deep networks by ensuring each sub-layer operates on a normalized input, allowing stable gradient flow to early layers.
- The trade-off is representation collapse: in deep Pre-Norm models, hidden states in successive layers become increasingly similar (high cosine similarity), diminishing the contribution of added layers.
- In the [[hyper-connections]] framework, Pre-Norm corresponds to the non-trainable HC matrix `[[0, 1], [1, 1]]` with `n=1` (Eq. 15).
- Most modern large language models (e.g., OLMo, GPT-style architectures) default to Pre-Norm for training stability.
- Empirically, Pre-Norm models trained without hyper-connections show median inter-layer cosine similarity approaching 1.0 in deeper layers, a signal of collapse.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[zhu-2025-hyperconnections-2409-19606]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[zhu-2025-hyperconnections-2409-19606]] as one pole of the normalization seesaw.
