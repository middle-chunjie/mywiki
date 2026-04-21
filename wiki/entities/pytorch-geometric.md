---
type: entity
title: PyTorch Geometric
slug: pytorch-geometric
date: 2026-04-20
entity_type: tool
aliases: [PyG, torch_geometric]
tags: [library, graph-learning]
---

## Description

PyTorch Geometric (PyG) is a graph deep learning library built on PyTorch, providing efficient implementations of GNN operators and pooling methods for graph-structured data. It is used in PGNN-EK to implement the GGNN message-passing layers.

## Key Contributions

- Provides graph batching, message-passing primitives, and pooling operators used to implement GGNN in PGNN-EK.
- Enables mini-batch training over graphs of variable size via automatic batching.

## Related Concepts

- [[graph-neural-network]]
- [[gated-graph-neural-network]]
- [[message-passing-neural-network]]

## Sources

- [[zhu-2022-neural]]
