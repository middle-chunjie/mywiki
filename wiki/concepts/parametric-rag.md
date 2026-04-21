---
type: concept
title: Parametric RAG
slug: parametric-rag
date: 2026-04-20
updated: 2026-04-20
aliases: [P-RAG, Parametric Retrieval Augmented Generation, 参数化检索增强生成]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Parametric RAG** (参数化检索增强生成) — a retrieval-augmented generation paradigm that injects retrieved document knowledge into model parameters, rather than relying only on prompt-context concatenation.

## Key Points

- The paper instantiates Parametric RAG as a Retrieve-Update-Generate pipeline, where retrieved document adapters are merged into the base LLM before answering.
- Each external document is preprocessed into a document-specific LoRA-style parameter block instead of being passed online as raw text.
- The method targets feed-forward network weights, arguing that parametric knowledge injection better matches how LLMs store internal knowledge.
- It improves over standard in-context RAG on multiple QA benchmarks and can be combined with in-context RAG for stronger results.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[su-2025-parametric-2501-15915]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[su-2025-parametric-2501-15915]].
