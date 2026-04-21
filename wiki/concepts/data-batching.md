---
type: concept
title: Data Batching
slug: data-batching
date: 2026-04-20
updated: 2026-04-20
aliases: [batch construction, memory-aware batching, 数据批次构建]
tags: [training, language-model, memory, retrieval]
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Data Batching** (数据批次构建) — the strategy of selecting and grouping training examples into a mini-batch, which in memory-augmented LM training directly determines what context-target pairs are available as in-batch memories and thus shapes the training-inference alignment.

## Key Points

- **Default (random) batching**: segments are randomly sampled from the corpus; supports only local memory training (TRIMELM).
- **Consecutive-segment batching**: `m > 1` consecutive segments from the same document are packed into one batch (TRIME notation: `B` total segments, `b ≈ B/m` distinct documents per batch). Enables back-propagation through long-term memory representations and training the model to leverage 15k–25k token context at inference.
- **BM25 batching**: segments with high lexical overlap (BM25 score) are greedily packed together. Serves as a training proxy for semantic nearest-neighbor retrieval at test time, closing the gap with large external datastores. Algorithm: start from a random segment, repeatedly append the highest-BM25-scoring available segment (`k = 20` candidates queried).
- Enabling gradient propagation to memory representations within the same batch is crucial; experiments without back-prop show significant perplexity degradation (45.15 → 41.50 on WIKITEXT-103 7M dev).
- Batch construction choice should match the target inference memory type; using a mismatched batching strategy degrades performance (Table 7, TRIME paper).

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[zhong-2022-training-2205-12674]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[zhong-2022-training-2205-12674]].
