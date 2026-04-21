---
type: concept
title: Semantic Chunking
slug: semantic-chunking
date: 2026-04-20
updated: 2026-04-20
aliases: [embedding-based chunking, semantic splitting, 语义切块]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Semantic Chunking** (语义切块) — splitting text by detecting semantic changes between neighboring units, typically with embedding similarity or distance signals.

## Key Points

- The paper treats Semantic Chunking as a strong baseline that better respects meaning than fixed character or separator rules.
- On narrative books, the authors argue Semantic Chunking can over-fragment dialogue-heavy passages because paragraph-level embedding changes are noisy.
- In GutenQA retrieval, Semantic Chunking trails LumberChunker across every reported `DCG@k` and `Recall@k` cutoff.
- The method yields average chunks of `185` tokens on the benchmark, much shorter than LumberChunker’s `334`-token average.
- Semantic Chunking is more parallelizable than LumberChunker because its boundary decisions do not depend on sequential LLM calls.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[duarte-2024-lumberchunker-2406-17526]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[duarte-2024-lumberchunker-2406-17526]].
