---
type: entity
title: LLM2Vec-Llama-3-8B-Supervised
slug: llm2vec-llama-3-8b-supervised
date: 2026-04-20
entity_type: model
aliases: [LLM2Vec-Llama-3-8B-Supervised]
tags: []
---

## Description

LLM2Vec-Llama-3-8B-Supervised is a pre-trained retriever checkpoint evaluated in [[unknown-nd-rare-2410-20088]] for exemplar-augmented fine-tuning. The paper uses it to test whether RARe also helps models that were already trained as text encoders rather than adapted directly from a generative LLM.

## Key Contributions

- Serves as one of the two main retriever-checkpoint baselines for RARe.
- Improves BeIR All from `55.35` to `56.76`, while showing only mixed changes on RAR-b (`23.44` to `23.10`), highlighting backbone sensitivity.

## Related Concepts

- [[dense-retrieval]]
- [[contrastive-learning]]
- [[domain-generalization]]

## Sources

- [[unknown-nd-rare-2410-20088]]
