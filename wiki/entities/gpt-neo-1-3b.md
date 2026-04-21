---
type: entity
title: GPT-Neo 1.3B
slug: gpt-neo-1-3b
date: 2026-04-20
entity_type: tool
aliases: [gpt-neo-1.3B, GPT-Neo-1.3B]
tags: []
---

## Description

GPT-Neo 1.3B is one of the strongest small generative models benchmarked in [[almeida-2024-exploring]] for zero-shot synthetic question generation. In beam-search mode it achieves the best high-quality-question configuration in the paper's main benchmark.

## Key Contributions

- Delivers the top `hitsR` configuration when paired with beam search.
- Produces synthetic data that trains a BERT reranker above BM25 on every reported dataset.

## Related Concepts

- [[small-language-model]]
- [[zero-shot-prompting]]
- [[question-generation]]
- [[beam-search]]

## Sources

- [[almeida-2024-exploring]]
