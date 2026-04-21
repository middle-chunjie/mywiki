---
type: concept
title: Adaptive Generation
slug: adaptive-generation
date: 2026-04-20
updated: 2026-04-20
aliases: [adaptive passage generation, 自适应生成]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Adaptive Generation** (自适应生成) — a generation strategy in which the model decides how much internal content to produce up to a specified maximum instead of always emitting a fixed number of passages.

## Key Points

- Astute RAG prompts the LLM to generate at most `\hat{m}` internal passages and to stop when no more reliable information is available.
- The design emphasizes reliability and coverage rather than diversity alone, using constitutional guidance in the prompt.
- In the main setup, `\hat{m} = 1`, which keeps the method cheap while still exposing useful internal knowledge.
- The generated passage count is genuinely adaptive: average `m` is `0.69` when `\hat{m} = 1` and `1.24` when `\hat{m} = 2`.
- Increasing `\hat{m}` from `1` to `2` yields a small additional accuracy gain, suggesting the mechanism does not over-generate by default.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[wang-2024-astute-2410-07176]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[wang-2024-astute-2410-07176]].
