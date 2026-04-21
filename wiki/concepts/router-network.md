---
type: concept
title: Router Network
slug: router-network
date: 2026-04-20
updated: 2026-04-20
aliases: [routing network, router, 路由网络]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Router Network** (路由网络) — a gating module that scores candidate experts or modules and dispatches each token representation to the selected computation path.

## Key Points

- Mixtral computes router logits with a learned linear map `xW_g` and applies `softmax(TopK(...))` over the selected experts.
- Each token is routed independently at every layer and time step, allowing different experts to be chosen across positions and layers.
- Mixtral sets `K = 2`, so the router produces a weighted combination of the top-`2` experts for each token.
- The paper's routing analysis finds little clean domain specialization; expert choices look more syntactic and position-local than topic-specific.
- Routing imbalance is highlighted as a practical systems issue for expert-parallel training and inference.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[jiang-2024-mixtral-2401-04088]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[jiang-2024-mixtral-2401-04088]].
