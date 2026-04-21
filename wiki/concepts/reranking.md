---
type: concept
title: Reranking
slug: reranking
date: 2026-04-20
updated: 2026-04-20
aliases: [re-ranking, 重排序]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Reranking** (重排序) — a post-processing step that reorders multiple candidate outputs using an auxiliary scoring signal to choose a better final prediction.

## Key Points

- [[geng-2024-large]] samples multiple Codex comments for the same prompt, then reranks them against a reference comment from the most similar retrieved code snippet.
- The paper evaluates both token-overlap and semantic-similarity rerankers.
- Reranking improves results even without better demonstration retrieval, though the gains are larger when combined with strong demonstrations.
- The best system in the paper combines semantic demonstration selection with token-based reranking.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[geng-2024-large]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[geng-2024-large]].
