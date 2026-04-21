---
type: concept
title: Category-aware Recommendation
slug: category-aware-recommendation
date: 2026-04-20
updated: 2026-04-20
aliases: [类别感知推荐, category-aware sequential recommendation]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Category-aware Recommendation** (类别感知推荐) — a recommendation approach that incorporates item category information as an explicit signal for modeling user intent and ranking candidate items.

## Key Points

- The paper argues that category information exposes higher-level user intent that item IDs alone cannot represent.
- HPM models category sequences explicitly rather than treating side information as a lightweight auxiliary feature.
- Category-aware contrastive learning is used to preserve category consistency and to distinguish different category-level intents.
- Prior side-information fusion approaches are discussed as insufficient on sparse datasets, motivating explicit category modeling.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[huang-2023-dual]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[huang-2023-dual]].
