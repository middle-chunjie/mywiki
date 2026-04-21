---
type: entity
title: monoT5
slug: mono-t5
date: 2026-04-20
entity_type: tool
aliases: [MonoT5, castorini/monot5-base-msmarco-10k]
tags: []
---

## Description

monoT5 is the pretrained sequence-to-sequence reranking model evaluated in [[almeida-2024-exploring]] as an alternative question-quality estimator. The authors find it competitive but ultimately prefer BM25 for filtering because BM25 is cheaper and more directly retrieval-oriented.

## Key Contributions

- Provides a relevance-model alternative to BM25 in the question quality filter.
- Acts as a strong external baseline in the downstream comparison table.

## Related Concepts

- [[question-quality-filtering]]
- [[neural-reranking]]
- [[information-retrieval]]

## Sources

- [[almeida-2024-exploring]]
