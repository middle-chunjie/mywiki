---
type: concept
title: Graph Convolution
slug: graph-convolution
date: 2026-04-20
updated: 2026-04-20
aliases: [图卷积, GCN propagation]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Graph Convolution** (图卷积) — a neighborhood aggregation operation that updates node representations by mixing each node's features with those of adjacent nodes according to graph structure.

## Key Points

- TGR applies parameter-free graph convolution to inject prerequisite structure into tool embeddings after dependency prediction.
- The update rule is `D^{-1/2}(A + I)D^{-1/2}X`, where `A` is the tool adjacency matrix and `X` is the initial tool embedding matrix.
- Self-loops via `A + I` preserve each tool's own representation while allowing information flow from dependent or prerequisite neighbors.
- The paper removes trainable GCN parameters to reduce latency during retrieval-time graph encoding.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[unknown-nd-tool-2508-05152]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[unknown-nd-tool-2508-05152]].
