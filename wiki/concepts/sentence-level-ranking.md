---
type: concept
title: Sentence-Level Ranking
slug: sentence-level-ranking
date: 2026-04-20
updated: 2026-04-20
aliases: [sentence ranking, 句子级排序]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Sentence-Level Ranking** (句子级排序) — ranking candidate sentences by their relevance to a query, often as a finer-grained alternative to passage-level retrieval.

## Key Points

- AGRaME scores a sentence by restricting MaxSim to the token embeddings belonging to that sentence while keeping passage-level encoding fixed.
- The paper introduces a new query marker `m'_q` specifically for sentence-level scoring so the query encoder can shift from passage identification to sentence discrimination.
- To rank sentences across passages, AGRaME adds the in-passage sentence score to a passage-level relevance score.
- Experiments show large sentence-level gains on Natural Questions, TriviaQA, WebQuestions, and EntityQuestions without materially hurting passage-level ranking.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[reddy-2024-agrame-2405-15028]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[reddy-2024-agrame-2405-15028]].
