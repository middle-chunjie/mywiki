---
type: concept
title: Generative ranking
slug: generative-ranking
date: 2026-04-20
updated: 2026-04-20
aliases: [autoregressive ranking, 生成式排序]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Generative ranking** (生成式排序) — a ranking approach that constructs the final order sequentially by selecting one item at each generation step instead of scoring the full list only once.

## Key Points

- GenRT uses sequence generation to emit the final reranked list with decreasing relevance step by step.
- Previously generated high-relevance documents act as sequential dependency signals for subsequent selection.
- This formulation lets the model align reranking decisions with step-local truncation decisions inside one inference process.
- The paper reports that the generative view improves separation between positive and negative candidates over steps compared with SetRank.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[xu-2024-listaware-2402-02764]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[xu-2024-listaware-2402-02764]].
