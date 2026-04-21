---
type: concept
title: Instance normalization
slug: instance-normalization
date: 2026-04-20
updated: 2026-04-20
aliases: [IN, 实例归一化]
tags: [normalization, representation-learning]
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Instance normalization** (实例归一化) — channel-wise normalization of feature maps using per-instance statistics to suppress style or domain-specific variation.

## Key Points

- FRPT uses parameter-free instance normalization inside CAH to reduce species-level discrepancies in frozen semantic features.
- The operation is applied per channel as `(M_P^i - E[M_P^i]) / sqrt(Var[M_P^i] + epsilon)`.
- The paper does not apply IN blindly; it gates the normalized features with channel attention to avoid erasing discriminative information.
- Ablation results indicate that using IN is better than simply discarding the supposedly irrelevant channels.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[wang-2023-finegrained-2207-14465]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[wang-2023-finegrained-2207-14465]].
