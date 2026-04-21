---
type: concept
title: Neural Ranking Model
slug: neural-ranking-model
date: 2026-04-20
updated: 2026-04-20
aliases: [神经排序模型, neural ranker]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Neural Ranking Model** (神经排序模型) — a ranking model that learns relevance scoring functions from neural representations or neural interaction features rather than relying only on hand-engineered retrieval features.

## Key Points

- The paper positions K-NRM as a neural ranking model tailored for ad-hoc retrieval rather than semantic matching in general.
- It contrasts representation-based neural rankers with interaction-based ones, arguing that word-level interaction signals are more appropriate for document ranking.
- K-NRM uses neural embeddings plus differentiable kernel pooling so relevance supervision directly shapes the similarity space.
- Empirically, K-NRM outperforms earlier neural baselines such as DRMM and CDSSM across all reported evaluation settings.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[xiong-2017-endtoend-1706-06613]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[xiong-2017-endtoend-1706-06613]].
