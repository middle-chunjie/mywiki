---
type: entity
title: nomic-bert-2048
slug: nomic-bert-2048
date: 2026-04-20
entity_type: tool
aliases: [nomic bert 2048]
tags: []
---

## Description

nomic-bert-2048 is the long-context encoder backbone trained in [[nussbaum-2025-nomic-2402-01613]] before contrastive stages. It adapts BERT with RoPE, SwiGLU, FlashAttention, and a `2048`-token training context.

## Key Contributions

- Serves as the initialization point for weakly supervised contrastive pretraining.
- Shows competitive GLUE average performance despite long-context architectural modifications.
- Enables later extrapolation to `8192` tokens through [[dynamic-ntk-scaling]].

## Related Concepts

- [[masked-language-modeling]]
- [[rotary-positional-embedding]]
- [[dynamic-ntk-scaling]]

## Sources

- [[nussbaum-2025-nomic-2402-01613]]
