---
type: concept
title: Tree Positional Encoding
slug: tree-positional-encoding
date: 2026-04-20
updated: 2026-04-20
aliases: [tree position encoding, 树位置编码]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Tree Positional Encoding** (树位置编码) — a positional encoding method that represents nodes by their location in a tree structure so attention layers can exploit ancestry, branching, and sibling relations rather than only linear order.

## Key Points

- The paper describes each AST node with a recursive list of 2D coordinates, where each coordinate stores sibling order and the parent's number of children.
- This description is designed to preserve more structural information than prior encodings based only on sibling order or shortest-path distance.
- The model splits tree positional encoding into a global absolute component derived from the full coordinate list and a local relative component defined on adjacent nodes.
- Coordinate lists are embedded, padded or truncated to a maximum depth, and then converted into attention biases instead of being treated as ordinary token embeddings.
- Across both completion and summarization tasks, the combined local-plus-global tree positional encoding outperforms earlier tree Transformer baselines.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[peng-nd-rethinking]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[peng-nd-rethinking]].
