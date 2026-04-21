---
type: entity
title: GenRT
slug: genrt
date: 2026-04-20
entity_type: tool
aliases: [Reranking-Truncation joint model]
tags: []
---

## Description

GenRT is the joint encoder-decoder model proposed in [[xu-2024-listaware-2402-02764]] for concurrent reranking and truncation in list-aware retrieval.

## Key Contributions

- Shares global list-level features between reranking and truncation in a single model.
- Generates the final ranking step by step while making cut decisions with a local backward window.
- Reaches the paper's best reported performance on both web-search and retrieval-augmented QA settings.

## Related Concepts

- [[list-aware-retrieval]]
- [[generative-ranking]]
- [[local-backward-window]]

## Sources

- [[xu-2024-listaware-2402-02764]]
