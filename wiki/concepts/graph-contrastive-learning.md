---
type: concept
title: Graph Contrastive Learning
slug: graph-contrastive-learning
date: 2026-04-20
updated: 2026-04-20
aliases: [GCL, 图对比学习]
tags: [graph-learning, self-supervised-learning]
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Graph Contrastive Learning** (图对比学习) — a self-supervised graph representation learning paradigm that learns invariant node or graph embeddings by maximizing agreement between multiple views of the same graph data.

## Key Points

- A typical pipeline contains graph data augmentation, paired view encoders, and a contrastive objective such as InfoNCE.
- MA-GCL argues that many prior GCL methods generate views that are too similar because augmentations are weak and encoders are architecturally identical.
- The paper reframes augmentation as architectural perturbation of the GNN encoder rather than only perturbation of graph inputs.
- In MA-GCL, asymmetric depth, random depth, and operator shuffling are combined as plug-and-play improvements to a simple GRACE-like backbone.
- Stronger GCL in this paper is operationalized by higher downstream node-classification accuracy together with lower mutual information between the two views.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[gong-2022-magcl-2212-07035]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[gong-2022-magcl-2212-07035]].
