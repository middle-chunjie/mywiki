---
type: concept
title: Personalization
slug: personalization
date: 2026-04-20
updated: 2026-04-20
aliases: [个性化, user personalization]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Personalization** (个性化) — the adaptation of a system's behavior or outputs to a specific user's preferences, history, or context.

## Key Points

- This paper studies personalization for frozen LLMs by retrieving personal documents from each user's profile instead of fine-tuning per-user model parameters.
- The user profile `P_u` is treated as the source of personal evidence, and personalization quality is judged by downstream task metrics rather than explicit relevance labels.
- Different inputs benefit from different personalization signals, including recency, lexical matching, semantic matching, and learned personalized retrievers.
- On LaMP, the best personalized pipeline improves over a non-personalized LLM on all seven tasks, with average gain reported as `15.3%`.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[salemi-2024-optimization]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[salemi-2024-optimization]].
