---
type: concept
title: Cross-graph Embedding Exchange
slug: cross-graph-embedding-exchange
date: 2026-04-20
updated: 2026-04-20
aliases: [跨图嵌入交换, CEE]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Cross-graph Embedding Exchange** (跨图嵌入交换) — a mechanism that swaps entity embeddings between original and cross-constructed graphs at each layer to accelerate semantic propagation across aligned knowledge graphs.

## Key Points

- RHGN first constructs cross graphs by exchanging seed-aligned entities between the two KGs.
- At layer `k + 1`, the original graph consumes cross-graph embeddings and the cross graph consumes original embeddings, creating bidirectional propagation.
- The paper argues this shortens the path through which evidence moves between KGs compared with simply adding alignment edges.
- Ablation results show removing CEE hurts alignment performance, indicating that neighbor heterogeneity benefits from explicit cross-graph exchange.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[liu-2023-rhgn]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[liu-2023-rhgn]].
