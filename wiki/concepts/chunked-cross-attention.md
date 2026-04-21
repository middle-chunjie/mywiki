---
type: concept
title: Chunked Cross-Attention
slug: chunked-cross-attention
date: 2026-04-20
updated: 2026-04-20
aliases: [CCA, 分块交叉注意力]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Chunked Cross-Attention** (分块交叉注意力) — an autoregressive cross-attention mechanism in which decoder states for one chunk attend only to encoded retrieval neighbors aligned to the preceding chunk.

## Key Points

- RETRO applies chunked cross-attention over `64`-token chunks so retrieval can be integrated without breaking left-to-right generation.
- The mechanism attends over both neighbor chunks and their continuations, with the neighbor and time axes flattened before attention.
- Relative positional encodings are added inside the cross-attention softmax to preserve alignment structure between retrieved text and current decoding states.
- The paper inserts this operator every `3` decoder layers starting from layer `6`, finding that this schedule is a better trade-off than using cross-attention only at the top, bottom, or all layers.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[borgeaud-2022-improving-2112-04426]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[borgeaud-2022-improving-2112-04426]].
