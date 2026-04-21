---
type: concept
title: Tree-Based Exploration
slug: tree-based-exploration
date: 2026-04-20
updated: 2026-04-20
aliases: [tree search exploration, 树式探索]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Tree-Based Exploration** (树式探索) — an inference strategy that expands multiple candidate improvement directions as tree branches instead of committing to a single linear reasoning path.

## Key Points

- SolutionRAG uses tree growth because the path from a weak draft solution to a reliable engineering plan is not fixed.
- Each node expands into `H` sampled proposals, letting the system consider different redesign or critique directions in parallel.
- The paper alternates between solution nodes and comment nodes, so exploration covers both drafting and reviewing behaviors.
- Pruning keeps only the highest-scoring nodes at each layer, making the search tractable.
- Ablation shows removing the tree structure lowers overall analytical / technical scores from `66.2/64.1` to `62.7/61.7`.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[li-2025-deepsolution-2502-20730]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[li-2025-deepsolution-2502-20730]].
