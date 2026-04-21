---
type: concept
title: Fusion Encoder
slug: fusion-encoder
date: 2026-04-20
updated: 2026-04-20
aliases: [融合编码器]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Fusion Encoder** (融合编码器) — an encoder module that explicitly fuses representations from multiple input streams, here document and query embeddings, through cross-attention to support fine-grained interaction.

## Key Points

- [[liu-2025-gear-2501-02772]] adds a fusion encoder as the central module that upgrades a standard bi-encoder into a model with local retrieval capability.
- The fusion encoder shares most parameters with the query encoder but inserts lightweight learnable cross-attention modules.
- It lets document embeddings from `E_d(.)` interact with query embeddings at each layer rather than only through a final scalar similarity.
- GeAR uses the fusion encoder's cross-attention weights at inference time to rank sentence-level evidence inside documents.
- This module supplies the fused representations consumed by the lightweight text decoder.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[liu-2025-gear-2501-02772]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[liu-2025-gear-2501-02772]].
