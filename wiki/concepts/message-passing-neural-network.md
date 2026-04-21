---
type: concept
title: Message Passing Neural Network
slug: message-passing-neural-network
date: 2026-04-20
updated: 2026-04-20
aliases: [MPNN, 消息传递神经网络]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Message Passing Neural Network** (消息传递神经网络) — a graph neural network that updates node states by repeatedly aggregating messages from neighboring nodes and then pooling them into graph-level representations.

## Key Points

- ASGN uses an MPGNN as the shared backbone for both the teacher and the student models.
- The paper defines node updates through a learnable aggregation function over neighbor messages and chooses sum aggregation in practice.
- Edge messages are modulated by Gaussian radial basis features derived from inter-atomic distances, making the model distance-aware.
- After `L` propagation layers, node embeddings are pooled into a graph embedding that feeds an MLP for molecular property prediction.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[hao-2020-asgn]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[hao-2020-asgn]].
