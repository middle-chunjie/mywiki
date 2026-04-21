---
type: concept
title: Unified Information Access
slug: unified-information-access
date: 2026-04-20
updated: 2026-04-20
aliases: [Universal Information Access, UIA]
tags: [information-retrieval, recommendation, dense-retrieval, personalization]
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Unified Information Access** (统一信息获取) — the paradigm of building a single model that can efficiently and effectively perform multiple distinct information access functionalities (e.g., keyword search, query by example, recommendation) without task-specific architectures.

## Key Points

- Treats information access as a generic scoring function `f(F, R, H, I; θ)` with four inputs: functionality, request, user history, and candidate item — enabling a single model to handle tasks with identical inputs but different retrieval semantics.
- Information access functionality `F` is encoded as natural-language text, making the framework extensible to new tasks without adding parameters.
- Joint optimization transfers knowledge across tasks, benefiting data-scarce tasks (e.g., complementary item recommendation) with up to 45% NDCG@10 relative gains over task-specific training.
- The UIA paper demonstrates that 60%+ of users benefit from joint optimization on query-by-example and complementary recommendation tasks.
- Motivated by Zamani & Croft's hypothesis that joint modeling of search and collaborative filtering can generalize better than separate models.

## My Position

<!-- User's stance on this concept. Fed by personal writing; tag "(personal stance)" on such bullets. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[zeng-2023-personalized-2304-13654]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[zeng-2023-personalized-2304-13654]].
