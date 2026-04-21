---
type: concept
title: Triplet Position
slug: triplet-position
date: 2026-04-20
updated: 2026-04-20
aliases: [triplet positional encoding]
tags: [ast, positional-information]
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Triplet Position** — a node-position scheme for abstract syntax trees that encodes node depth, parent-layer width, and sibling order so hierarchical syntax remains distinguishable during representation learning.

## Key Points

- [[guo-2022-modeling]] assigns each AST node a triplet `{depth, parent-width, sibling-width}` instead of flattening the tree into a sequence or path.
- Function nodes use non-negative sibling-width indices, while attribute nodes use negative indices so the model can separate control nodes from value-bearing leaves.
- Width positions are computed by left-to-right breadth traversal, and the root node is fixed at `{0, 0, 0}`.
- The paper uses these triplets as learnable positional embeddings added to AST node embeddings before GraphSAGE encoding.
- Removing triplet positions lowers Java performance from `49.19/32.27/59.59` to `48.53/31.62/58.84` on BLEU/METEOR/ROUGE-L.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[guo-2022-modeling]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[guo-2022-modeling]].
