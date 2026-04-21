---
type: concept
title: Tree of Reasoning
slug: tree-of-reasoning
date: 2026-04-20
updated: 2026-04-20
aliases: [ToR]
tags: [reasoning, retrieval]
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Tree of Reasoning** — a tree-structured search space whose branches are regenerated reasoning chains, allowing retrieval feedback to redirect the model to corrected or completed subproblems.

## Key Points

- In SearChain, every generated Chain-of-Query becomes a branch of the Tree-of-Reasoning rather than a final immutable chain.
- Retrieval verification and completion identify the node that should be corrected or expanded, and the next Chain-of-Query is regenerated from that node.
- The paper describes this process as node-identify depth-first search, contrasting it with standard depth-first backtracking and with one-dimensional reasoning chains.
- The correct path through the tree is traced at the end to produce final content with step-level supporting references.
- This tree view is central to the claim that SearChain can dynamically modify reasoning direction when the current branch is inconsistent or underspecified.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[xu-2024-searchinthechain-2304-14732]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[xu-2024-searchinthechain-2304-14732]].
