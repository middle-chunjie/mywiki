---
type: concept
title: Chunk Representation
slug: chunk-representation
date: 2026-04-20
updated: 2026-04-20
aliases: [Chunk Representation, 分块表示]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Chunk Representation** (分块表示) — a condensed vector representation of a chunk that summarizes the chunk's content for downstream modeling.

## Key Points

- [[li-2024-chulo-2410-11119]] constructs each chunk representation as a weighted average of token embeddings rather than by keeping all token states.
- Keyphrase tokens receive weight `a` and non-keyphrase tokens receive weight `b`, yielding `c = (Σ_t w_t * t) / (Σ_t w_t)` with `a > b`.
- The representation is designed to preserve semantic salience after document compression and then serve as input to a chunk-level Transformer module.
- Ablation results suggest that adding sentence embeddings to this representation hurts performance instead of helping.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[li-2024-chulo-2410-11119]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[li-2024-chulo-2410-11119]].
