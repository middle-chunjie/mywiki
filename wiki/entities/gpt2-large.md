---
type: entity
title: GPT2-large
slug: gpt2-large
date: 2026-04-20
entity_type: model
aliases: [GPT-2 Large, GPT2 Large, GPT2-l]
tags: []
---

## Description

GPT2-large is the small language model used in [[wang-2024-large-2301-11916]] to learn concept-token embeddings and score candidate demonstrations before transferring the selected examples to larger models. In the paper, it functions as the inexpensive selector model rather than the final deployment model.

## Key Contributions

- Learns task-specific concept tokens used to approximate latent task variables.
- Produces transferable demonstration rankings that improve average 4-shot accuracy across larger GPT-family models.
- Serves as an evaluation model itself, where the method reaches `64.8` average accuracy across eight classification datasets.

## Related Concepts

- [[prompt-tuning]]
- [[demonstration-selection]]
- [[in-context-learning]]

## Sources

- [[wang-2024-large-2301-11916]]
