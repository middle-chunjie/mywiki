---
type: concept
title: Context Relevance
slug: context-relevance
date: 2026-04-20
updated: 2026-04-20
aliases: [context focus]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Context Relevance** (上下文相关性) — the degree to which retrieved context is focused on the information needed to answer a question, with minimal redundant material.

## Key Points

- RAGAS treats context relevance as a property of retrieval focus, not just retrieval recall.
- The evaluator LLM extracts the subset of sentences in the context that are needed to answer the question.
- The score is `CR = extracted_sentences / total_sentences`, so verbose or noisy contexts are penalized.
- Context relevance is the hardest of the three reported RAGAS dimensions, reaching `0.70` agreement with human judgments on WikiEval.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[es-2023-ragas-2309-15217]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[es-2023-ragas-2309-15217]].
