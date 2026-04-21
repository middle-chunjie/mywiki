---
type: concept
title: Dynamic Retrieval-Augmented Generation
slug: dynamic-retrieval-augmented-generation
date: 2026-04-20
updated: 2026-04-20
aliases: [dynamic RAG, 动态检索增强生成]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Dynamic Retrieval-Augmented Generation** (动态检索增强生成) — a retrieval-augmented decoding paradigm that decides during generation when retrieval should be triggered and what query should be sent to the retriever.

## Key Points

- [[su-2024-dragin-2403-10081]] frames dynamic RAG as a two-part control problem: choosing retrieval timing and constructing the retrieval query at each trigger point.
- The paper argues fixed-token and fixed-sentence schedules can over-retrieve, adding noise and unnecessary inference cost.
- DRAGIN replaces static scheduling with token-level signals derived from uncertainty and self-attention, making retrieval contingent on the model's current information needs.
- Across four QA benchmarks, the paper reports dynamic retrieval with better control yielding stronger results than both single-round RAG and prior dynamic heuristics.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[su-2024-dragin-2403-10081]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[su-2024-dragin-2403-10081]].
