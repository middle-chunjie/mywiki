---
type: concept
title: Many-Shot In-Context Learning
slug: many-shot-in-context-learning
date: 2026-04-20
updated: 2026-04-20
aliases: [many-shot-icl]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Many-Shot In-Context Learning** (多示例上下文学习) — a regime where a model receives a large number of labeled demonstrations in the prompt and must infer a new task from those examples without parameter updates.

## Key Points

- HELMET treats many-shot ICL as a distinct long-context capability rather than assuming it correlates with recall-oriented tasks.
- The benchmark uses high-label-cardinality datasets such as TREC coarse/fine, NLU, BANKING77, and CLINC150.
- To reduce leakage from pretrained label semantics, the paper remaps natural-language labels into numeric IDs like `0`, `1`, and `2`.
- The paper finds that ICL trends often differ from retrieval and citation tasks, reinforcing the need for diverse evaluation axes.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[yen-2024-helmet-2410-02694]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[yen-2024-helmet-2410-02694]].
