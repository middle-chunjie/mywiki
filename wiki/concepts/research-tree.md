---
type: concept
title: Research Tree
slug: research-tree
date: 2026-04-20
updated: 2026-04-20
aliases: [research tree, 研究树]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Research Tree** (研究树) — a tree-structured representation of a deep-research problem in which vertices denote entities or facts and edges encode the dependencies used to derive the final answer.

## Key Points

- InfoSeek builds each synthesized problem around a research tree whose root is the final answer and whose internal vertices become intermediate sub-problems.
- A root can be initialized from a sampled entity page, then expanded recursively by attaching child entities or factual constraints.
- Parent blurring adds multiple child claims so that intermediate vertices correspond to valid constraint satisfaction problems rather than trivial identifiers.
- Vertical expansion through dependency links increases tree depth and makes the final problem require additional reasoning hops.
- Question generation happens only after the tree reaches the desired complexity and all vertices have sufficient supporting constraints.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[xia-2025-open-2509-00375]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[xia-2025-open-2509-00375]].
