---
type: concept
title: Graph Augmentation
slug: graph-augmentation
date: 2026-04-20
updated: 2026-04-20
aliases: [graph data augmentation, 图增强]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Graph Augmentation** (图增强) — the process of transforming a graph's structure or attributes to generate alternative views for training representation models, especially in self-supervised learning.

## Key Points

- MSSGCL argues that graph augmentation is not automatically semantics-preserving, because different augmented subgraphs can encode meaning at different scales.
- The paper contrasts node dropping, attribute masking, edge perturbation, and subgraph sampling as common augmentation strategies in graph contrastive learning.
- It specializes augmentation into two scale-aware distributions, one for global views and one for local views.
- The main methodological claim is that augmentation choice should determine the contrastive objective, not just the input perturbation.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[liu-2023-multiscale]]
- [[jiang-2024-ragraph-2410-23855]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[liu-2023-multiscale]].
