---
type: concept
title: Sequential Recommendation
slug: sequential-recommendation
date: 2026-04-20
updated: 2026-04-20
aliases: [序列推荐, next-item recommendation]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Sequential Recommendation** (序列推荐) — the task of predicting the next item a user is likely to interact with from an ordered history of past interactions.

## Key Points

- The paper formulates each user context as `O_i = {V_i, C_i, T_i}`, combining item IDs, item categories, and timestamps for next-item prediction.
- It argues that accurate sequential recommendation requires modeling both rapidly changing item-level preference and more stable category-level preference.
- The evaluation follows leave-one-out next-item prediction with `99` sampled negatives and metrics `HR@K` and `NDCG@K` for `K in {5, 10, 20, 50}`.
- HPM consistently outperforms strong sequential recommendation baselines including FPMC, GRU4Rec, Caser, SASRec, TiSASRec, Chorus, and KDA.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[huang-2023-dual]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[huang-2023-dual]].
