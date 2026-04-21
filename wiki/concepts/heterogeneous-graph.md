---
type: concept
title: Heterogeneous Graph
slug: heterogeneous-graph
date: 2026-04-20
updated: 2026-04-20
aliases: [异构图]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Heterogeneous Graph** (异构图) — a graph containing multiple node or edge types so that structurally different entities and relations can be modeled within one representation space.

## Key Points

- [[jin-2023-code]] models users, files, repositories, and directory structures as typed nodes connected by typed relations.
- CODER uses a hierarchical heterogeneous graph for project structure and separate bipartite graphs for file-level and project-level behaviors.
- This design lets the model bridge microscopic contribution behavior with macroscopic repository preference signals.
- The heterogeneous setup is necessary because OSS interactions are not well described by a flat user-item graph alone.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[jin-2023-code]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[jin-2023-code]].
