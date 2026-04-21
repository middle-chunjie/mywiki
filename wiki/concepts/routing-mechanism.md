---
type: concept
title: Routing Mechanism
slug: routing-mechanism
date: 2026-04-20
updated: 2026-04-20
aliases: [routing, router, 路由机制]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Routing Mechanism** (路由机制) — a procedure that selects which specialized model, expert, or computational path should process a given input instance.

## Key Points

- RouterRetriever uses a training-free routing mechanism built on embedding similarity rather than a learned classifier head.
- The router compares a base-encoder query embedding against a library of pilot embeddings and averages scores by expert before making one expert choice.
- This retrieval-specific design outperforms classifier-based and dataset-sampling routers on BEIR.
- The paper's analysis shows routing is still imperfect, because RouterRetriever remains much closer to DatasetOracle than to InstanceOracle.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[lee-2024-routerretriever-2409-02685]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[lee-2024-routerretriever-2409-02685]].
