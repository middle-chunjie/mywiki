---
type: entity
title: Tok-RAG
slug: tok-rag
date: 2026-04-20
entity_type: tool
aliases: [Tok RAG]
tags: []
---

## Description

Tok-RAG is the token-level collaborative decoding method proposed in [[xu-2025-theory-2406-00944]]. It runs pure-LLM and RAG generation in parallel and selects tokens by comparing similarity to retrieved-text and pure-LLM representations.

## Key Contributions

- Turns the paper's benefit-versus-detriment theory into an inference-time decoding rule without extra training.
- Improves robustness to noisy retrieval across short-form QA, long-form QA, dialogue, code generation, slot filling, and language modeling.
- Achieves higher token-level benefit-detriment classification performance than hallucination-based baselines across OPT, LLaMA-2, and Mistral.

## Related Concepts

- [[retrieval-augmented-generation]]
- [[distribution-fusion]]
- [[token-level-generation]]
- [[cosine-similarity]]

## Sources

- [[xu-2025-theory-2406-00944]]
