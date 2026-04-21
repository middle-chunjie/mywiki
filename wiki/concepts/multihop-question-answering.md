---
type: concept
title: Multihop Question Answering
slug: multihop-question-answering
date: 2026-04-20
updated: 2026-04-20
aliases: [multi-hop QA, 多跳问答]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Multihop Question Answering** (多跳问答) — question answering that requires combining evidence from multiple facts, documents, or reasoning steps before producing the final answer.

## Key Points

- [[su-2024-dragin-2403-10081]] evaluates dynamic RAG on [[2wiki-multihopqa]] and [[hotpotqa]] as its primary multihop benchmarks.
- For these datasets, the models generate both chain-of-thought style reasoning traces and final answers, then are scored with EM and F1.
- The paper reports its strongest relative gains on multihop datasets, supporting the claim that dynamic retrieval is especially useful when intermediate reasoning reveals new evidence needs.
- The retrieval setup uses BM25 over Wikipedia with `top-k = 3`, showing that better retrieval control alone can improve multihop QA without retraining the base LLM.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[su-2024-dragin-2403-10081]]
- [[guti-rrez-2024-hipporag-2405-14831]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[su-2024-dragin-2403-10081]].
