---
type: concept
title: Forum Answer Ranking
slug: forum-answer-ranking
date: 2026-04-20
updated: 2026-04-20
aliases: [论坛答案排序, answer ranking]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Forum Answer Ranking** (论坛答案排序) — the task of ordering answers to a developer question by estimated quality or usefulness.

## Key Points

- BLANCA defines this task over `500K` questions, each with at least `3` answers.
- Answer popularity ranks are normalized to scores in `[0, 1]` and learned with cosine similarity loss.
- Evaluation uses information-retrieval metrics `MRR` and `NDCG`.
- Multi-task BERTOverflow gives the best reported performance, exceeding both untuned and single-task tuned baselines.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[abdelaziz-2022-can]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[abdelaziz-2022-can]].
