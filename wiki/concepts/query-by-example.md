---
type: concept
title: Query by Example
slug: query-by-example
date: 2026-04-20
updated: 2026-04-20
aliases: [QBE]
tags: [information-retrieval, recommendation, retrieval]
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Query by Example** (示例查询) — an information access paradigm where the user specifies an anchor item (rather than a text query) and the system retrieves items similar to that anchor.

## Key Points

- Input is the textual content of an anchor item; the system ranks candidate items by semantic similarity to the anchor rather than relevance to a keyword query.
- Closely related to item-based collaborative filtering but distinguishes between similar-item retrieval and complementary-item recommendation despite sharing identical input structure.
- In the UIA framework, QBE and complementary item recommendation receive the same `(R, F, H, I)` inputs; only the functionality description `F` differentiates them.
- Task-specific training for QBE is prone to over-fitting via lexical overlap, causing the model to retrieve the exact anchor item as the top result; joint training with other tasks regularizes this.
- Among the three information access tasks in UIA, QBE benefits most from cross-task training (+34% NDCG@10 with joint optimization over task-specific training).

## My Position

<!-- User's stance on this concept. Fed by personal writing; tag "(personal stance)" on such bullets. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[zeng-2023-personalized-2304-13654]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[zeng-2023-personalized-2304-13654]].
