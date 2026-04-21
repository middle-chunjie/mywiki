---
type: concept
title: Gated Graph Neural Network
slug: gated-graph-neural-network
date: 2026-04-20
updated: 2026-04-20
aliases: [GGNN, 门控图神经网络]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Gated Graph Neural Network** (门控图神经网络) — a graph neural network that iteratively propagates edge-typed messages over graph neighborhoods and updates node states with recurrent gating, typically a GRU.

## Key Points

- [[long-2022-multiview]] uses one GGNN encoder per graph view rather than one monolithic graph encoder.
- Node states are initialized from one-hot node-type vectors `h_u^0 = x_u`.
- Message passing is edge-type dependent through `m_{u,v}^t = f_e(h_v^{t-1})`, followed by mean aggregation and a GRU update.
- The final representation of each view is obtained by max pooling node states after `T` propagation steps.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[long-2022-multiview]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[long-2022-multiview]].
