---
type: concept
title: Answer Relevance
slug: answer-relevance
date: 2026-04-20
updated: 2026-04-20
aliases: [answer relevancy]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Answer Relevance** (答案相关性) — the degree to which a generated answer directly addresses the user question without being incomplete or padded with unnecessary information.

## Key Points

- RAGAS evaluates answer relevance independently from factuality; a response can be relevant yet unfaithful, or faithful yet incomplete.
- The metric generates reverse questions from the answer and compares them to the original question in embedding space.
- The formal score is the mean cosine similarity `AR = (1 / n) * sum_i sim(q, q_i)` over generated reverse questions.
- On WikiEval, answer relevance reaches `0.78` agreement with human judgments, lower than faithfulness because many pairwise differences are subtle.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[es-2023-ragas-2309-15217]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[es-2023-ragas-2309-15217]].
