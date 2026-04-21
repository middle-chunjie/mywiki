---
type: entity
title: TREC DL 2020
slug: trec-dl-2020
date: 2026-04-20
entity_type: dataset
aliases: [TREC Deep Learning 2020]
tags: []
---

## Description

TREC DL 2020 is the dense deep-learning retrieval benchmark used in [[qin-2024-large-2306-17563]] to evaluate zero-shot passage reranking. The paper reports results on `54` queries reranked from the BM25 top-`100` over the MS MARCO passage corpus.

## Key Contributions

- Serves as the strongest benchmark in the paper, where FLAN-UL2-based PRP reaches up to `85.80` at `NDCG@1`.
- Provides dense relevance labels for comparing PRP against supervised rerankers and GPT-based listwise prompting baselines.

## Related Concepts

- [[document-reranking]]
- [[bm25]]
- [[passage-relevance]]

## Sources

- [[qin-2024-large-2306-17563]]
