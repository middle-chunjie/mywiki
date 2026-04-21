---
type: entity
title: Natural Questions
slug: natural-questions
date: 2026-04-20
entity_type: dataset
aliases: [NQ]
tags: []
---

## Description

Natural Questions is a large open-domain question answering and retrieval dataset used in [[almeida-2024-exploring]] for validation and downstream reranking experiments. The paper treats it as a benchmark for synthetic question generation in a collection with more than `2M` documents.

## Key Contributions

- Supplies one of the five evaluation settings for both NI validation and reranker benchmarking.
- Shows substantial gains over BM25 after training on synthetic data, including `0.416` nDCG@10 for the GPT-Neo 1.3B beam-search configuration.

## Related Concepts

- [[information-retrieval]]
- [[question-generation]]
- [[neural-reranking]]

## Sources

- [[almeida-2024-exploring]]
