---
type: concept
title: Iterative Query Generation
slug: iterative-query-generation
date: 2026-04-20
updated: 2026-04-20
aliases: [LLM-free query generation, next-hop query construction]
tags: [rag, retrieval, multi-hop-qa]
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Iterative Query Generation** (迭代查询生成) — the process of automatically constructing follow-up retrieval queries from intermediate retrieval results across multiple rounds, used in multi-hop RAG to progressively gather all evidence needed to answer a complex question.

## Key Points

- Traditional iterative RAG methods generate new queries by calling an LLM at each retrieval round, incurring high latency and cost; EfficientRAG replaces LLM calls with a lightweight token-level classifier (the Filter).
- The Filter takes the concatenation of the current query and labeled informative tokens from `<Continue>`-tagged chunks, then extracts verbatim tokens to form the next-hop query — no new tokens are introduced.
- This approach reduces the number of LLM calls per query to 1 (only the final answer generator) versus 3–7 for LLM-based iterative methods, achieving ~3x latency reduction.
- Training data for the Filter is synthesized by prompting Llama-3-70B-Instruct to generate next-hop questions given sub-question dependencies, then applying the same token-extraction procedure used at inference.
- Out-of-domain experiments show the approach generalizes across datasets (HotpotQA ↔ 2WikiMQA), suggesting it captures domain-independent relational patterns.

## My Position

<!-- User's stance on this concept. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[zhuang-2024-efficientrag-2408-04259]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[zhuang-2024-efficientrag-2408-04259]].
