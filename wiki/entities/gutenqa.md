---
type: entity
title: GutenQA
slug: gutenqa
date: 2026-04-20
entity_type: dataset
aliases: [GutenQA]
tags: []
---

## Description

GutenQA is the benchmark dataset introduced by the LumberChunker paper. It contains 100 manually extracted Project Gutenberg narrative books paired with 3,000 highly specific question-answer pairs for evaluating retrieval and RAG chunking quality.

## Key Contributions

- Supplies the main evaluation benchmark for LumberChunker.
- Measures passage retrieval with `DCG@k` and `Recall@k` over `3000` question-answer pairs.
- Targets narrative retrieval rather than encyclopedic fact lookup, making chunk coherence especially important.

## Related Concepts

- [[benchmark]]
- [[passage-retrieval]]
- [[question-generation]]

## Sources

- [[duarte-2024-lumberchunker-2406-17526]]
