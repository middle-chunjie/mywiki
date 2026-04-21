---
type: concept
title: Session Graph
slug: session-graph
date: 2026-04-20
updated: 2026-04-20
aliases: [会话图]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Session Graph** (会话图) — a graph representation of a search session in which queries are nodes and typed inter-query relations encode how turns are connected.

## Key Points

- [[mao-2022-convtrans]] reorganizes each raw web search session into a heterogeneous graph instead of preserving the original linear order.
- The graph uses three edge types: response-induced, topic-shared, and topic-changed.
- Response-induced edges are detected when a later query overlaps heavily with a sentence from an earlier clicked passage, while topic-shared edges rely on lexical overlap between queries.
- To improve graph quality, the method expands candidate neighbors from the whole session database and then keeps only the Top-5 response-induced and Top-5 topic-shared edges per central node.
- The session graph is the structural substrate from which pseudo conversations are later sampled by a constrained random walk.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[mao-2022-convtrans]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[mao-2022-convtrans]].
