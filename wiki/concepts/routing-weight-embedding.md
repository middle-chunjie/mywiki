---
type: concept
title: Routing-Weight Embedding
slug: routing-weight-embedding
date: 2026-04-20
updated: 2026-04-20
aliases: [RW embedding, 路由权重嵌入]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Routing-Weight Embedding** (路由权重嵌入) — an embedding built from the per-layer router probability distributions of an MoE model rather than from its hidden activations alone.

## Key Points

- The paper defines `e_RW` by concatenating the routing probability vectors from all MoE layers.
- Last-token routing weights outperform averaging routing weights across all tokens on STS benchmarks.
- RW embeddings reveal cluster structures and topic groupings that differ substantially from hidden-state embeddings.
- RW is especially strong at capturing paraphrastic and high-level semantic similarity, and it complements HS rather than replacing it.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[li-2024-your-2410-10814]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[li-2024-your-2410-10814]].
