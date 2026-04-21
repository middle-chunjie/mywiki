---
type: concept
title: Text Embedding
slug: text-embedding
date: 2026-04-20
updated: 2026-04-20
aliases: [text embedding, text embeddings, 文本嵌入]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Text Embedding** (文本嵌入) — a vector representation of text intended to preserve semantic content so that similarity, retrieval, clustering, or classification can be done geometrically in embedding space.

## Key Points

- The paper targets both token-level and sequence-level embeddings rather than only generation quality.
- It evaluates text embeddings on MTEB as well as linear probing for chunking, NER, and POS tagging.
- The central claim is that decoder-only LLMs can become strong text embedders once the causal masking mismatch is corrected.
- The best LLM2Vec systems outperform strong encoder-only baselines on word-level tasks and set new unsupervised MTEB records at the time.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[behnamghader-2024-llmvec-2404-05961]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[behnamghader-2024-llmvec-2404-05961]].
