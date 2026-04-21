---
type: concept
title: Linear Attention
slug: linear-attention
date: 2026-04-20
updated: 2026-04-20
aliases: [线性注意力]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Linear Attention** (线性注意力) — an attention variant whose computation is arranged so sequence processing scales linearly with sequence length instead of quadratically as in standard softmax self-attention.

## Key Points

- The paper shows that one-step TTT with linear `f, g, h, phi, psi`, `W_0 = 0`, and `eta = 1` is mathematically equivalent to linear attention.
- Under this derivation, `theta_phi`, `theta_psi`, and `theta_g` play the roles of key, query, and value projections, while the learned inner-loop weights `W_1` act like the attention state.
- The equivalence is empirically supported on ImageNet patches: linear attention with identity map scores `73.0%`, while `MTTT-Linear` scores `72.8%`.
- On long sequences such as raw pixels, linear attention remains runnable while standard self-attention becomes memory- and FLOP-prohibitive, which motivates the paper's MTTT-MLP extension.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[sun-2024-learn-2310-13807]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[sun-2024-learn-2310-13807]].
