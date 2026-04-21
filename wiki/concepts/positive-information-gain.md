---
type: concept
title: Positive Information Gain
slug: positive-information-gain
date: 2026-04-20
updated: 2026-04-20
aliases: []
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Positive Information Gain** — the property that a RAG model's output improves upon retrieved input by becoming more direct, more accurate, or more complete.

## Key Points

- The paper uses positive information gain to characterize the intended behavior of an LLM across three retrieval scenarios with different evidence quality.
- In Scenario 1, gain means compressing complex retrieved text into the directly useful continuation.
- In Scenario 2, gain means correcting wrong facts and completing missing facts inside corrupted retrieved text.
- In Scenario 3, gain means using semantically related context to stimulate relevant parametric knowledge even when retrieved passages do not contain the answer.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[xu-2024-unsupervised-2402-18150]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[xu-2024-unsupervised-2402-18150]].
