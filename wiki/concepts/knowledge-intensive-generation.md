---
type: concept
title: Knowledge-Intensive Generation
slug: knowledge-intensive-generation
date: 2026-04-20
updated: 2026-04-20
aliases: [knowledge intensive generation, 知识密集型生成]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Knowledge-Intensive Generation** (知识密集型生成) — generation tasks in which model outputs depend on external factual knowledge or evidence that is not reliably contained in parametric model memory alone.

## Key Points

- [[su-2024-dragin-2403-10081]] uses 2WikiMultihopQA, HotpotQA, StrategyQA, and IIRC as representative knowledge-intensive generation benchmarks.
- The paper argues that single-round retrieval often misses information needs that emerge only after partial reasoning or partial generation.
- DRAGIN shows larger gains on multihop datasets than on StrategyQA, suggesting retrieval control matters most when the task requires stepwise evidence acquisition.
- Wikipedia segmented into `100`-token passages serves as the external knowledge source in the paper's experiments.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[su-2024-dragin-2403-10081]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[su-2024-dragin-2403-10081]].
