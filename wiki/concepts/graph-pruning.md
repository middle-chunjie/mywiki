---
type: concept
title: Graph Pruning
slug: graph-pruning
date: 2026-04-20
updated: 2026-04-20
aliases: [graph pruning, tree pruning, context pruning]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Graph Pruning** (图剪枝) — a structure-aware filtering operation that removes low-utility branches from a graph or tree before downstream retrieval, reasoning, or generation.

## Key Points

- The paper prunes one child subtree from a retrieved code graph and keeps the variant with the highest cosine similarity to the query.
- Pruning is applied before prompt construction, so it acts as a context-budget control mechanism rather than a post hoc cleanup step.
- In the proposed PKG pipeline, pruning is defined over AST-derived containment DAGs rooted at retrieved function or block nodes.
- The motivation is to reduce irrelevant retrieved code while preserving enough syntactic context for code generation.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[unknown-nd-contextaugmented-2601-20810]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[unknown-nd-contextaugmented-2601-20810]].
