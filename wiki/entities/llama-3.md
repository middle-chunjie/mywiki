---
type: entity
title: Llama 3
slug: llama-3
date: 2026-04-20
entity_type: model
aliases: [Llama-3]
tags: []
---

## Description

Llama 3 is the decoder-only base model family used in [[unknown-nd-rare-2410-20088]] for retrieval training from language-model checkpoints. The paper tests both plain Llama-3 and Llama-3.1-Instruct to measure whether RARe can preserve or recover in-context learning benefits in embedding-style retrieval.

## Key Contributions

- Provides the backbone for RARe experiments trained from LLM checkpoints.
- Supports gains from `38.68` to `40.99` average nDCG@10 on Llama-3 and from `39.94` to `40.88` against Promptriever on Llama-3.1-Instruct.

## Related Concepts

- [[in-context-learning]]
- [[dense-retrieval]]
- [[contrastive-learning]]

## Sources

- [[unknown-nd-rare-2410-20088]]
