---
type: concept
title: Reactive Retrieval
slug: reactive-retrieval
date: 2026-04-20
updated: 2026-04-20
aliases:
  - 响应式检索
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Reactive Retrieval** (响应式检索) — a retrieval setting in which the system only returns documents after an explicit trigger from the user or interface.

## Key Points

- In ProCIS, reactive contextual suggestion assumes the request happens at the end of the observed conversation.
- The full conversation history is used as the query representation for ranking Wikipedia articles.
- Evaluation follows standard ranking metrics such as `nDCG@k`, `MRR`, `MAP`, and `Recall@k`.
- LMGR substantially outperforms lexical and neural baselines in the paper's reactive benchmark.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[samarinas-2024-procis]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[samarinas-2024-procis]].
