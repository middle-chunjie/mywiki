---
type: entity
title: MetaRAG
slug: metarag
date: 2026-04-20
entity_type: system
aliases: [Metacognitive RAG, MetaRAG framework]
tags: [rag, metacognition, multi-hop-qa]
---

## Description

MetaRAG is a retrieval-augmented generation framework introduced by Zhou et al. (2024) that augments a standard QA LLM with a metacognition space consisting of a monitoring stage, an evaluating stage, and a planning stage. The system uses a fine-tuned T5-large expert monitor and GPT-3.5-turbo-16k as the base QA and evaluator-critic model.

## Key Contributions

- First framework to explicitly operationalize metacognitive regulation (monitor–evaluate–plan) within the RAG pipeline for multi-hop QA.
- Achieves EM of 42.8 on 2WikiMultihopQA and 37.8 on HotpotQA, outperforming Reflexion by 34.6% and 26.0% respectively.
- Code released at https://github.com/ignorejjj/MetaRAG.

## Related Concepts

- [[metacognitive-retrieval-augmented-generation]]
- [[metacognitive-regulation]]
- [[retrieval-augmented-generation]]
- [[knowledge-conflict]]
- [[multihop-question-answering]]

## Sources

- [[zhou-2024-metacognitive-2402-11626]]
