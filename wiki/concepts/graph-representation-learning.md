---
type: concept
title: Graph Representation Learning
slug: graph-representation-learning
date: 2026-04-20
updated: 2026-04-20
aliases: [graph-level representation learning, 图表示学习]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Graph Representation Learning** (图表示学习) — the process of learning vector representations for entire graphs so their structure, attributes, and task-relevant regularities can be used by downstream models.

## Key Points

- ASGN explicitly learns graph-level embeddings for molecules in addition to node embeddings.
- The paper assumes that similar molecules in chemical space should have similar properties, so graph representations should reflect dataset-level structure.
- It introduces a clustering-based self-supervised loss over graph embeddings to encourage globally meaningful partitions.
- These graph embeddings are reused both for property prediction and for diversity-based active selection.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[hao-2020-asgn]]
- [[liu-2023-multiscale]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[hao-2020-asgn]].
