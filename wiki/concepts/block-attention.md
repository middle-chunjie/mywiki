---
type: concept
title: Block Attention
slug: block-attention
date: 2026-04-20
updated: 2026-04-20
aliases: [block attention, 块注意力]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Block Attention** (块注意力) — an attention mechanism that partitions an input prompt into semantically independent blocks, computes most blocks independently, and lets only designated blocks attend globally so block-level KV states can be reused across prompts.

## Key Points

- The paper applies block attention to RAG by assigning each retrieved passage to one block and treating the user query as the final globally attending block.
- When a cached passage block changes, only the modified block and the final block need recomputation, avoiding full re-encoding of all downstream tokens.
- Block attention is paired with position re-encoding so a cached block can be relocated to a new absolute prompt position without recomputing its internal KV states from scratch.
- The method requires block-aware fine-tuning; a direct switch from full attention to block attention causes large accuracy degradation.
- After adaptation, the same model can switch between block attention and full attention with little or no performance loss on the reported benchmarks.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[unknown-nd-blockattention-2409-15355]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[unknown-nd-blockattention-2409-15355]].
