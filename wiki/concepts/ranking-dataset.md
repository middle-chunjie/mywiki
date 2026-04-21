---
type: concept
title: Ranking Dataset
slug: ranking-dataset
date: 2026-04-20
updated: 2026-04-20
aliases: [排序数据集]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Ranking Dataset** (排序数据集) — a collection of queries, documents or passages, and labels used to train or evaluate retrieval and reranking systems.

## Key Points

- DL-MIA is built as a derivative ranking dataset over TREC-DL 2021 and 2022 rather than a standalone corpus.
- Its basic supervision unit is a tuple `(query, intent, passage, label)` instead of the usual `(query, passage, label)` formulation.
- The final release contains `2655` labeled tuples over `24` queries and `69` finalized intents.
- The paper positions the dataset as reusable for reranking, diversification, intent coverage, and query suggestion experiments.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[anand-2024-understanding-2408-17103]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[anand-2024-understanding-2408-17103]].
