---
type: concept
title: Hierarchical Preference Modeling
slug: hierarchical-preference-modeling
date: 2026-04-20
updated: 2026-04-20
aliases: [层级偏好建模, multi-granular preference modeling]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Hierarchical Preference Modeling** (层级偏好建模) — a recommendation strategy that separately models user preference dynamics at multiple granularities, such as fine-grained items and coarse-grained categories.

## Key Points

- The paper treats item-ID preference as fast-changing and category preference as relatively stable, and models them with separate sequence encoders.
- HPM uses average-pooled item representation `v_f` and category representation `c_f` as two complementary views of user state.
- The authors argue that relying only on item IDs misses coarse-grained intent and mischaracterizes preference drift over time.
- Their results suggest explicit hierarchical modeling is especially useful on sparse datasets such as Clothing.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[huang-2023-dual]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[huang-2023-dual]].
