---
type: entity
title: Mistral 7B
slug: mistral-7b
date: 2026-04-20
entity_type: tool
aliases:
  - Mistral-7B
  - Mistral 7B base model
tags: []
---

## Description

Mistral 7B is the decoder-only LLM backbone that [[lee-2024-nvembed-2405-17428]] converts into NV-Embed by removing the causal mask and adding latent-attention pooling.

## Key Contributions

- Serves as the initialization point for the paper's two-stage LoRA fine-tuning recipe.
- Provides the hidden states that are pooled by the latent-attention block into sequence embeddings.

## Related Concepts

- [[large-language-model]]
- [[bidirectional-attention]]
- [[latent-attention]]

## Sources

- [[lee-2024-nvembed-2405-17428]]
