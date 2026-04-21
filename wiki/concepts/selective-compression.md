---
type: concept
title: Selective Compression
slug: selective-compression
date: 2026-04-20
updated: 2026-04-20
aliases: [选择性压缩]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Selective Compression** (选择性压缩) — a context compression strategy that allocates different compression rates to different parts of an input according to estimated task relevance or importance.

## Key Points

- FlexRAG applies selective compression after encoding retrieved context into compressive embeddings, rather than removing raw text spans directly.
- The paper studies both token-level likelihood-based importance estimation and sentence-level embedding-based importance estimation.
- The compressed budget is assigned by grouping context into different priority tiers and solving an allocation relation of the form `w_1 * n_1 + ... + w_k * n_k = α * n`.
- In the reported experiments, sentence-level selective compression is the strongest variant, beating uniform down-sampling at the same overall `8x` budget.
- Different high-priority / low-priority allocations change the tradeoff between preserving crucial evidence and retaining broader background context.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[liu-2024-lighter-2409-15699]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[liu-2024-lighter-2409-15699]].
