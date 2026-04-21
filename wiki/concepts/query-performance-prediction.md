---
type: concept
title: Query Performance Prediction
slug: query-performance-prediction
date: 2026-04-20
updated: 2026-04-20
aliases: [QPP, 查询性能预测]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Query Performance Prediction** (查询性能预测) — the estimation of how effective a retrieval run is likely to be for a given query without access to ground-truth relevance labels.

## Key Points

- The paper uses unsupervised QPP methods such as `WIG`, `NQC`, `sigma_max`, and `sigma_50%` as baselines for choosing among candidate retrievers.
- QPP is relevant because the best retrieval strategy varies by input, but the personalization setting lacks query-document relevance annotations.
- Supervised QPP methods are ruled out in this work precisely because the required relevance labels do not exist for LaMP.
- Learned retriever selectors outperform the QPP baselines on nearly all datasets, especially in the post-generation setting.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[salemi-2024-optimization]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[salemi-2024-optimization]].
