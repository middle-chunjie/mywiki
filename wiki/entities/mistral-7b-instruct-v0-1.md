---
type: entity
title: Mistral-7B-Instruct-v0.1
slug: mistral-7b-instruct-v0-1
date: 2026-04-20
entity_type: model
aliases: [Mistral 7B Instruct v0.1, "mistralai/Mistral-7B-Instruct-v0.1"]
tags: []
---

## Description

Mistral-7B-Instruct-v0.1 is the main decoder-only backbone used in [[springer-2024-repetition-2402-15449]] for both zero-shot and fine-tuned embedding experiments. It is the model on which echo embeddings show the clearest gains over classical pooling baselines.

## Key Contributions

- Serves as the backbone for the headline zero-shot MTEB result of `48.64` with echo embeddings.
- Serves as the backbone for the main fine-tuned result of `64.68` average MTEB.
- Provides the strongest comparison point against causal and bidirectional classical embeddings in the paper.

## Related Concepts

- [[autoregressive-language-model]]
- [[echo-embedding]]
- [[text-embedding]]

## Sources

- [[springer-2024-repetition-2402-15449]]
