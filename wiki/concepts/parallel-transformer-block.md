---
type: concept
title: Parallel Transformer Block
slug: parallel-transformer-block
date: 2026-04-20
updated: 2026-04-20
aliases: [PTB, Parallel Block, 并行Transformer块]
tags: [neural-architecture, transformer, efficiency]
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Parallel Transformer Block** (并行Transformer块) — a Transformer architectural variant in which the attention sub-layer and the feed-forward network (FFN) sub-layer within a block process the same input independently and their outputs are summed, rather than feeding the attention output sequentially into the FFN.

## Key Points

- In the standard sequential arrangement, the FFN input depends on the attention output; in PTB, both sub-layers read the same input, enabling parallel computation.
- PTB is a special degenerate case of [[hyper-connections]] with `n=2`: specific HC matrix values (Eq. 18 in the paper) reproduce the PTB connectivity pattern.
- [[hyper-connections]] can spontaneously learn PTB-like patterns during training: analysis of the learned OLMo-1B-DHC×4 connection matrix reveals jagged (PTB-style) patterns at certain layer pairs (e.g., layers 11 and 12), suggesting the model finds PTB arrangements beneficial for some depths.
- PTB reduces sequential depth of the transformer by combining two sub-layers into one effective step, which can improve inference latency on hardware with sufficient parallel compute.
- Within the [[sequential-parallel-duality]] framework, PTB and standard sequential blocks are two extremes; DHC learns a continuous mixture between them, dynamically per token.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[zhu-2025-hyperconnections-2409-19606]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[zhu-2025-hyperconnections-2409-19606]] as a special case of hyper-connections and an emergent learned pattern.
