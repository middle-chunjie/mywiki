---
type: concept
title: LLM Reranker
slug: llm-reranker
date: 2026-04-20
updated: 2026-04-20
aliases: [large language model reranker, 大语言模型重排序器]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**LLM Reranker** (大语言模型重排序器) — a reranking system that uses a large language model to rescore an initial candidate set by reading richer document and context information than the first-stage retriever uses.

## Key Points

- STaRK reranks the top candidates returned by `ada-002` rather than retrieving directly with the LLM.
- The paper uses GPT-4 Turbo and Claude 3 Opus to assign satisfaction scores in `[0, 1]` to each candidate.
- Reranking is applied with `k = 20` for synthesized queries and `k = 10` for human-generated queries because of cost.
- Rerankers improve top-rank effectiveness on semi-structured retrieval but still leave large accuracy gaps and add roughly `25s` to `26s` average latency.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[wu-2024-stark-2404-13207]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[wu-2024-stark-2404-13207]].
