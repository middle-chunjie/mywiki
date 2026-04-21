---
type: concept
title: Semantics-enhanced Context Embedding Learning
slug: semantics-enhanced-context-embedding-learning
date: 2026-04-20
updated: 2026-04-20
aliases: [语义增强上下文嵌入学习, SCEL]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Semantics-enhanced Context Embedding Learning** (语义增强上下文嵌入学习) — a context encoding method that enriches target representations with explicit relation signals and their time-varying influence from historical interactions.

## Key Points

- SCEL uses relation embeddings to connect history items with the target through explicit semantics such as `also_buy` and `also_view`.
- Complementary relations are modeled with Gaussian decay around `Delta t = 0`, while substitute relations combine short-term negative and long-term positive kernels.
- The enhanced target embeddings `e_{v,n}` and `e_{c,n}` are added to the original target item and category embeddings before scoring and contrastive learning.
- The ablation without SCEL shows the largest degradation, especially on sparse datasets, indicating that relation-aware context is a major contributor.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[huang-2023-dual]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[huang-2023-dual]].
