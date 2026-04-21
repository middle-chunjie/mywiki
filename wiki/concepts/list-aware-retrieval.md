---
type: concept
title: List-aware retrieval
slug: list-aware-retrieval
date: 2026-04-20
updated: 2026-04-20
aliases: [listwise retrieval, 列表感知检索]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**List-aware retrieval** (列表感知检索) — a retrieval setting that models the ranked list jointly, using cross-document context rather than scoring each document independently.

## Key Points

- The paper frames list-aware retrieval as covering both reranking and truncation over the same ranked list.
- GenRT argues that reranking and truncation should share list-level contextual features instead of being optimized in separate stages.
- In this formulation, the ranked list is the modeling unit for both relevance refinement and cut-off prediction.
- The motivation is especially strong for web search and retrieval-augmented generation, where users or LLMs consume an ordered list rather than isolated documents.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[xu-2024-listaware-2402-02764]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[xu-2024-listaware-2402-02764]].
