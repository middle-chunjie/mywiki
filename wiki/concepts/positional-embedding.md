---
type: concept
title: Positional Embedding
slug: positional-embedding
date: 2026-04-20
updated: 2026-04-20
aliases: [position embedding, positional embeddings, 位置嵌入]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Positional Embedding** (位置嵌入) — a representation attached to token positions so a Transformer can encode order information that token embeddings alone do not provide.

## Key Points

- [[ratner-2023-parallel-2212-10947]] reuses the first `C` trained positional embeddings across all parallel context windows instead of extrapolating to unseen positions.
- The task suffix keeps the final `T` positional slots, so generated outputs see every window as equally close in the positional sense.
- The paper applies the same high-level idea to both learned positional embeddings and rotary positional embeddings.
- Reusing position IDs without also changing the attention mask would be unsafe because tokens sharing a position were not trained to attend to each other.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[ratner-2023-parallel-2212-10947]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[ratner-2023-parallel-2212-10947]].
