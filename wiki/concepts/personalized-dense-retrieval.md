---
type: concept
title: Personalized Dense Retrieval
slug: personalized-dense-retrieval
date: 2026-04-20
updated: 2026-04-20
aliases: []
tags: [dense-retrieval, personalization, information-retrieval]
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Personalized Dense Retrieval** (个性化稠密检索) — the extension of bi-encoder dense retrieval models to incorporate user-specific preferences and interaction history, producing per-user request representations that adapt the retrieval output to individual needs.

## Key Points

- Extends the standard bi-encoder framework by augmenting the query/request vector with personalization signals derived from the user's past interactions (content-based) and user/task ID embeddings (collaborative).
- Requires a two-stage training: non-personalized pre-training on aggregated data (to handle cold-start users), followed by personalized fine-tuning on users with sufficient interaction history (≥10 interactions in UIA).
- Personalization is applied only on the request side; item representations are computed offline and indexed, preserving retrieval efficiency via ANN search.
- Effective for e-commerce retrieval and recommendation; the UIA framework reports 93–145% NDCG@10 improvement over non-personalized dense retrieval via APN.
- Not all users benefit: ~25% see performance degradation, particularly cold-start users or those with intent drift.

## My Position

<!-- User's stance on this concept. Fed by personal writing; tag "(personal stance)" on such bullets. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[zeng-2023-personalized-2304-13654]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[zeng-2023-personalized-2304-13654]].
