---
type: concept
title: Bayesian Personalized Ranking
slug: bayesian-personalized-ranking
date: 2026-04-20
updated: 2026-04-20
aliases: [贝叶斯个性化排序, BPR]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Bayesian Personalized Ranking** (贝叶斯个性化排序) — a pairwise ranking objective that learns to score observed items above sampled unobserved items for implicit-feedback recommendation.

## Key Points

- HPM optimizes recommendation quality with BPR loss `L_rec = -sum log sigma(y_hat_ui - y_hat_uj)` over positive and negative items.
- The positive score and negative score each combine item-level and category-level preference terms through dot products with enhanced target embeddings.
- The final optimization is multitask: `L_joint = L_rec + lambda L_cl`, so BPR provides the ranking backbone while DCL regularizes representation learning.
- The paper uses `1` negative sample during training and `99` sampled negatives for evaluation.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[huang-2023-dual]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[huang-2023-dual]].
