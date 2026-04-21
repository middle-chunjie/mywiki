---
type: concept
title: Predictive Reranking
slug: predictive-reranking
date: 2026-04-20
updated: 2026-04-20
aliases: [LM-dedicated reranking, 预测式重排序]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Predictive reranking** (预测式重排序) — a reranking strategy that selects retrieved documents according to how helpful they are for predicting the upcoming continuation of the current prefix.

## Key Points

- [[ram-2023-incontext-2302-00083]] trains a RoBERTa-base reranker over the top `k = 16` BM25 candidates returned for each prefix.
- The reranker defines `p_rank(d_i | x)` with a softmax over document scores and is optimized against LM likelihood on the next stride of tokens.
- Training uses `300,000` WikiText-103 examples, `10,000` steps, learning rate `1e-5`, and batch size `32`, while keeping the generator LM frozen.
- On WikiText-103, predictive reranking improves GPT-2 Small from `29.6` to `26.8` perplexity and GPT-2 XL from `16.6` to `15.4`, outperforming zero-shot reranking.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[ram-2023-incontext-2302-00083]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[ram-2023-incontext-2302-00083]].
