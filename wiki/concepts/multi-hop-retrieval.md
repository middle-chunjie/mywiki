---
type: concept
title: Multi-Hop Retrieval
slug: multi-hop-retrieval
date: 2026-04-20
updated: 2026-04-20
aliases: [multi-round retrieval, iterative multi-hop retrieval]
tags: [retrieval, rag, multi-hop-qa, information-retrieval]
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Multi-Hop Retrieval** (多跳检索) — a retrieval strategy that performs multiple sequential retrieval rounds where each round's query is derived from evidence gathered in prior rounds, enabling a system to collect all supporting documents needed to answer questions whose answer depends on chaining two or more facts.

## Key Points

- A single retrieval round fails for multi-hop questions (e.g., "What is the size of the shopping mall where radio station KGOT has its studios?") because the initial query lacks the bridging entity needed to find the final document.
- Iterative methods like Iter-RetGen, IRCoT, and SelfAsk implement multi-hop retrieval by alternating retrieval and reasoning steps, but each require one or more LLM calls per hop for query rewriting.
- EfficientRAG achieves comparable recall to LLM-based multi-hop retrieval methods while retrieving far fewer chunks (avg 6.41 on HotpotQA vs. 16–35 for competitors) by using lightweight token-level classifiers for query construction.
- The number of retrieval rounds is bounded by a maximum iteration count; early stopping occurs when all candidate chunks are tagged `<Terminate>`.
- Empirically, EfficientRAG requires ~2.73 retrieval iterations on MuSiQue while maintaining 1 LLM call total per question.

## My Position

<!-- User's stance on this concept. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[zhuang-2024-efficientrag-2408-04259]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[zhuang-2024-efficientrag-2408-04259]].
