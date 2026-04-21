---
type: concept
title: Hierarchical Transformer
slug: hierarchical-transformer
date: 2026-04-20
updated: 2026-04-20
aliases: [hierarchical transformer, 层次化 Transformer]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Hierarchical Transformer** (层次化 Transformer) — a transformer architecture that processes a sequence at multiple granularities, typically combining coarse global representations with finer local representations or decoding.

## Key Points

- Block Transformer instantiates hierarchy with block-level global processing at lower layers and token-level local decoding at upper layers.
- The model first aggregates `L_B` tokens into one block embedding, then autoregressively models dependencies across blocks before decoding tokens inside the next block.
- This hierarchy reduces effective context length in the global module from `L` to `L / L_B`, which lowers self-attention and KV-cache costs.
- The paper argues that both hierarchical levels matter: the best trade-off for `L_B = 4` appears near a `1:1` parameter allocation between block and token decoders.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[ho-2024-block-2406-02657]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[ho-2024-block-2406-02657]].
