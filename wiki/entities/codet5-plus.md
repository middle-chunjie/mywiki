---
type: entity
title: CodeT5+
slug: codet5-plus
date: 2026-04-20
entity_type: tool
aliases: [CodeT5 Plus]
tags: []
---

## Description

CodeT5+ is the open family of code large language models introduced in [[wang-2023-codet-2305-07922]] by [[salesforce-research]]. It uses a flexible encoder-decoder design that supports code understanding, generation, retrieval, and retrieval-augmented generation.

## Key Contributions

- Combines span denoising, causal LM, contrastive learning, and matching in a two-stage pretraining recipe.
- Scales efficiently to `2B`, `6B`, and `16B` models by freezing pretrained CodeGen decoders and training only a shallow encoder plus cross-attention.
- Delivers strong open-model results on HumanEval, text-to-code retrieval, code completion, and math programming.

## Related Concepts

- [[encoder-decoder-architecture]]
- [[contrastive-learning]]
- [[retrieval-augmented-generation]]

## Sources

- [[wang-2023-codet-2305-07922]]
