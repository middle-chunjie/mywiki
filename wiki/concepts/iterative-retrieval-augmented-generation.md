---
type: concept
title: Iterative Retrieval-Augmented Generation
slug: iterative-retrieval-augmented-generation
date: 2026-04-20
updated: 2026-04-20
aliases: [iterative RaLM, iterative RAG serving, 迭代检索增强生成]
tags: [rag, retrieval, serving, generation]
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Iterative Retrieval-Augmented Generation** (迭代检索增强生成) — a class of RAG methods that interleave multiple retrieval steps with autoregressive generation within a single request, allowing the model to query the knowledge base repeatedly as context evolves.

## Key Points

- Contrasts with one-shot RAG, which retrieves once before generation begins; iterative methods query the corpus at regular intervals (e.g., every `k` tokens) or upon model-detected need.
- Examples include In-Context RALM (retrieval every 4 tokens), FLARE (active retrieval on low-confidence tokens), and kNN-LM (retrieval every token).
- Higher generation quality than one-shot RAG due to access to dynamically relevant documents, at the cost of `O(T/k)` retrieval calls for a response of length `T`.
- The frequent retrieval–generation interleaving is the primary serving bottleneck, motivating acceleration approaches such as [[speculative-retrieval]].
- Different from the self-evaluative iterative RAG pattern (where retrieval is driven by an answer-sufficiency judge); here interaction frequency is fixed or triggered by decoding confidence.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[zhang-2024-accelerating-2401-14021]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[zhang-2024-accelerating-2401-14021]].
