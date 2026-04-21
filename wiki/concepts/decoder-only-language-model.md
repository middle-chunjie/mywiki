---
type: concept
title: Decoder-Only Language Model
slug: decoder-only-language-model
date: 2026-04-20
updated: 2026-04-20
aliases: [decoder-only LM, decoder-only language model, causal language model, 仅解码器语言模型]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Decoder-Only Language Model** (仅解码器语言模型) — a language model architecture that predicts tokens autoregressively under a causal attention mask, so each position can access only left context during standard inference and training.

## Key Points

- The paper starts from decoder-only backbones such as S-LLaMA-1.3B, LLaMA-2-7B, Mistral-7B, and Meta-Llama-3-8B.
- Their causal masking is identified as a core obstacle for high-quality token and sentence embeddings.
- LLM2Vec keeps the pretrained decoder-only weights and modifies the attention mask instead of replacing the architecture.
- The paper argues these models remain attractive because of sample efficiency, tooling maturity, and instruction-following behavior.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[behnamghader-2024-llmvec-2404-05961]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[behnamghader-2024-llmvec-2404-05961]].
