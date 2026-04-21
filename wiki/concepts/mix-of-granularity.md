---
type: concept
title: Mix-of-Granularity
slug: mix-of-granularity
date: 2026-04-20
updated: 2026-04-20
aliases: [MoG, MoGG, Mix-of-Granularity-Graph, 混合粒度检索]
tags: [retrieval, rag, chunking, routing]
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Mix-of-Granularity** (混合粒度检索) — a RAG retrieval method that uses a learned router (MLP over RoBERTa query embeddings) to dynamically select the optimal chunk-granularity level from `n_gra` pre-indexed granularities, rather than committing to a single fixed chunk size.

## Key Points

- Inspired by [[mixture-of-experts]]: a router predicts a weight vector `w` over `n_gra` granularity levels; the finest-grained similarity scores are multiplied by `w` and the top-k chunk candidates are selected before resolving to the parent chunk at the optimal granularity `g_r = argmax_g w_g`.
- **Soft-label training**: because top-k selection is non-differentiable, the router is trained with soft labels built offline via BM25 + TF-IDF/RoBERTa/hitrate similarity; the label assigns 0.8 to the best granularity and 0.2 to the runner-up, training with binary cross-entropy.
- **MoGG extension**: documents are pre-processed into a graph (nodes = short sentence-level snippets; edges via BM25 similarity threshold); granularity is redefined as hopping range, enabling retrieval of thematically related but textually distant passages.
- On MedCorp, MoG improves over MedRAG by ~5% on average and ~8.7% vs CoT on smaller LLMs (GLM3, Qwen1.5); MoGG with only the Textbooks subset outperforms MoG trained on all of MedCorp in terms of improvement margin over MedRAG.
- Storage overhead is approximately 2.7× the original corpus (one corpus copy + 5 embedding sets); inference time increases ~60% due to the router module but scales marginally with additional granularity levels.

## My Position

<!-- User's stance on this concept. Fed by personal writing; tag "(personal stance)" on such bullets. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[zhong-2024-mixofgranularity-2406-00456]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[zhong-2024-mixofgranularity-2406-00456]].
