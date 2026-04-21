---
type: concept
title: Metacognitive Retrieval-Augmented Generation
slug: metacognitive-retrieval-augmented-generation
date: 2026-04-20
updated: 2026-04-20
aliases: [MetaRAG, metacognitive RAG, 元认知检索增强生成]
tags: [rag, metacognition, llm, multi-hop-qa]
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Metacognitive Retrieval-Augmented Generation** (元认知检索增强生成) — a RAG framework that adds a metacognition space alongside the standard QA cognition space, allowing the model to monitor answer quality, evaluate failure causes, and plan targeted corrections in an iterative loop.

## Key Points

- MetaRAG identifies three root causes of incorrect answers in multi-hop RAG: insufficient knowledge, conflicting knowledge, and erroneous reasoning — each addressed by a distinct planning strategy.
- The metacognition space is implemented as a separate evaluator-critic LLM role that operates on top of the base QA LLM, avoiding the need to modify model weights.
- Monitoring uses a lightweight fine-tuned T5-large expert model and cosine similarity threshold (`k = 0.4`) to gate whether metacognitive evaluation is needed, preserving efficiency for straightforward questions.
- The framework outperforms Reflexion — the closest self-critic baseline — by 26–34.6% EM on HotpotQA and 2WikiMultihopQA, with gains largest in conflicting-knowledge and sufficient-knowledge scenarios.
- Metacognition is most beneficial for complex questions; applying it universally (`k = 0.8`) does not maximize performance, mirroring human intuition-based reasoning for simple tasks.
- Code available at https://github.com/ignorejjj/MetaRAG.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[zhou-2024-metacognitive-2402-11626]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[zhou-2024-metacognitive-2402-11626]].
