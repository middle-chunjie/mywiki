---
type: concept
title: Data Sparsity
slug: data-sparsity
date: 2026-04-20
updated: 2026-04-20
aliases: [数据稀疏性, sparse feedback]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Data Sparsity** (数据稀疏性) — the regime in which observed supervision is extremely limited relative to the full user-item space, making preference estimation and generalization difficult.

## Key Points

- The paper frames sparse implicit feedback as a core failure mode for collaborative filtering and GNN recommenders.
- DCCF uses contrastive self-supervision to supplement sparse observed interactions with auxiliary training signals.
- Dataset densities are very low, including `4.0e-4` on Gowalla and `3.7e-4` on Amazon-book.
- User-group and item-group analysis shows DCCF improves results for inactive users and low-degree items compared with DGCF, DGCL, and LightGCN.
- The authors argue that disentangled global context helps propagate useful signals beyond direct local graph connectivity.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[ren-2023-disentangled]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[ren-2023-disentangled]].
