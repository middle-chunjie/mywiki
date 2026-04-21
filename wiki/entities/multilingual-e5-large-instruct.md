---
type: entity
title: Multilingual-E5-Large-Instruct
slug: multilingual-e5-large-instruct
date: 2026-04-20
entity_type: tool
aliases: [E5-Large-Instruct, Multilingual E5 Large Instruct, mE5-large-instruct]
tags: []
---

## Description

Multilingual-E5-Large-Instruct is an encoder-based multilingual dense embedding model used in [[zhou-2026-retrieve-2604-04949]] as the second retriever backbone for LRAT fine-tuning experiments, complementing the decoder-based Qwen3-Embedding-0.6B.

## Key Contributions

- Serves as the encoder-based retrieval baseline; LRAT training consistently improves its evidence recall and task success rate across all six agent backbones evaluated.
- Demonstrates that LRAT's training paradigm generalizes across different retriever architectures (encoder vs. decoder).

## Related Concepts

- [[dense-retrieval]]
- [[bi-encoder]]
- [[multilingual-retrieval]]

## Sources

- [[zhou-2026-retrieve-2604-04949]]
