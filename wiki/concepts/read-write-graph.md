---
type: concept
title: Read-Write Graph
slug: read-write-graph
date: 2026-04-20
updated: 2026-04-20
aliases: [RWG, read write graph, 读写图]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Read-Write Graph** (读写图) — a bipartite program graph that links operands and operations with directed read and write edges to capture how computations consume and produce values.

## Key Points

- [[long-2022-multiview]] introduces RWG to make operand-operation interactions explicit instead of leaving them implicit inside other views.
- RWG contains two node types: operands and operations.
- `Read` edges point from operands to operations to denote consumed values.
- `Write` edges point from operations to operands to denote produced results.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[long-2022-multiview]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[long-2022-multiview]].
