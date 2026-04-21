---
type: concept
title: Rubric-Based Relevance Scoring
slug: rubric-based-relevance-scoring
date: 2026-04-20
updated: 2026-04-20
aliases: [relevance rubric scoring, rubric-guided relevance scoring]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Rubric-Based Relevance Scoring** (基于评分细则的相关性打分) — a relevance estimation scheme that prompts a model with explicit score bands and task-specific criteria so it can produce interpretable absolute relevance scores instead of opaque logits or probabilities.

## Key Points

- Retro* defines a rubric with a task-specific relevance definition plus five score intervals over `0-100`.
- For each query-document pair, the model generates both a reasoning trajectory and an integer score, not only a binary or relative judgment.
- The score bands are meant to have direct semantic meaning, such as distinguishing highly relevant from slightly relevant evidence.
- The paper uses this scoring mechanism to support both threshold-based relevance measurement and downstream reranking.
- The authors show cleaner positive-vs-negative score separation than prior pointwise baselines such as RankLLaMA and Rank1.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[lan-2026-retro-2509-24869]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[lan-2026-retro-2509-24869]].
