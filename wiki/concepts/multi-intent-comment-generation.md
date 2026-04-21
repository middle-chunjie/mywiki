---
type: concept
title: Multi-Intent Comment Generation
slug: multi-intent-comment-generation
date: 2026-04-20
updated: 2026-04-20
aliases: [multi intent comment generation, 多意图注释生成]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Multi-Intent Comment Generation** (多意图注释生成) — the task of generating code comments tailored to different developer intents such as functionality, rationale, usage, implementation details, or properties.

## Key Points

- [[geng-2024-large]] frames the task as moving from a one-to-one code-to-comment mapping to a one-to-many mapping conditioned on intent.
- The paper evaluates five explicit intent categories: `what`, `why`, `how-to-use`, `how-it-is-done`, and `property`.
- It shows that few-shot prompting with a code LLM can outperform the prior supervised baseline DOME on this task.
- Retrieval of intent-matched demonstrations and reranking of sampled outputs both materially improve performance.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[geng-2024-large]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[geng-2024-large]].
