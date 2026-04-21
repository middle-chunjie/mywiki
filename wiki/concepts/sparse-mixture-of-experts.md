---
type: concept
title: Sparse Mixture-of-Experts
slug: sparse-mixture-of-experts
date: 2026-04-20
updated: 2026-04-20
aliases: [sparse MoE, SMoE, 稀疏专家混合]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Sparse Mixture-of-Experts** (稀疏专家混合) — a mixture-of-experts architecture that activates only a small top-k subset of experts for each token, separating total parameter count from per-token compute.

## Key Points

- Mixtral 8x7B uses an SMoE design with `8` experts per layer but activates only `2` experts for each token.
- The architecture exposes about `47B` total parameters while using only `13B` active parameters during inference.
- In Mixtral, every Transformer FFN block is replaced by an MoE block rather than only a subset of layers.
- The paper frames SMoE as the mechanism that lets an open-weights model match or exceed dense `70B` baselines at much lower active compute.
- The systems discussion notes that SMoE still creates routing and load-balancing challenges even when arithmetic cost is reduced.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[jiang-2024-mixtral-2401-04088]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[jiang-2024-mixtral-2401-04088]].
