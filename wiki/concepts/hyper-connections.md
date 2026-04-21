---
type: concept
title: Hyper-Connections
slug: hyper-connections
date: 2026-04-20
updated: 2026-04-20
aliases: [HC, Dynamic Hyper-Connections, DHC, Static Hyper-Connections, SHC, 超连接]
tags: [neural-architecture, residual-connections, transformer]
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Hyper-Connections** (超连接) — a learnable generalization of residual connections that replaces the fixed scalar skip-add with a matrix `HC ∈ R^{(n+1)×(n+1)}` encoding both depth-connections (weighted blending of layer output into `n` hidden copies) and width-connections (lateral information exchange among hidden copies), enabling a network to autonomously learn connection strengths rather than inheriting Pre-Norm or Post-Norm trade-offs.

## Key Points

- The core construction expands the hidden state into `n` copies (hyper hidden matrix `H ∈ R^{n×d}`); the HC matrix `A_m` selects a weighted combination as the single layer input, while `A_r` and `B` route the layer output back to each copy.
- Pre-Norm and Post-Norm residual connections are both expressible as non-trainable `n=1` HC matrices (Eq. 15–16 in the paper), confirming HC strictly generalizes both.
- With `n=1`, HC provides no performance gain; `n≥2` is necessary for width-connections to function, with `n=4` identified as the empirical sweet spot.
- Dynamic Hyper-Connections (DHC) make all HC weights input-dependent via a lightweight `norm → linear → tanh → scale` transform per layer, at a cost under 0.04% in parameter count and 0.2% in FLOPs.
- Learned DHC weight matrices exhibit a Λ-shaped dense connection pattern: Post-Norm-style local decay combined with Pre-Norm-style persistent access to low-layer representations; parallel transformer block patterns also emerge spontaneously.
- Width-connections (`WC`) are the most critical component: ablating `WC` degrades V2 eval loss by `+0.021` in OLMo-1B experiments; ablating `B` has a smaller effect.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[zhu-2025-hyperconnections-2409-19606]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[zhu-2025-hyperconnections-2409-19606]].
