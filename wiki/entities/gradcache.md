---
type: entity
title: GradCache
slug: gradcache
date: 2026-04-20
entity_type: tool
aliases: [Gradient Cache, Grad Cache]
tags: []
---

## Description

GradCache is the code release associated with [[gao-2021-scaling-2101-06983]] for training contrastive encoders with cached representation gradients. It operationalizes the paper's graph-less forward, representation-gradient caching, and sub-batch replay procedure.

## Key Contributions

- Provides an implementation of the paper's exact large-batch training method under memory limits.
- Supports practical integration of [[gradient-cache]] into [[dense-retrieval]] systems such as [[dpr]].

## Related Concepts

- [[gradient-cache]]
- [[contrastive-learning]]
- [[dense-retrieval]]

## Sources

- [[gao-2021-scaling-2101-06983]]
