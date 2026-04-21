---
type: concept
title: Weighted Maximum Satisfiability
slug: weighted-maximum-satisfiability
date: 2026-04-20
updated: 2026-04-20
aliases: [MAX-SMT, weighted max-sat, 加权最大可满足性]
tags: [formal-methods, optimization]
source_count: 1
confidence: low
domain_volatility: low
last_reviewed: 2026-04-20
---

## Definition

**Weighted Maximum Satisfiability** (加权最大可满足性) - an optimization problem that finds a satisfiable subset of weighted soft constraints while preserving all hard constraints.

## Key Points

- The paper encodes child-language static checks as soft clauses over AST nodes and solves for the largest satisfiable subset.
- Clauses omitted from the optimal solution identify AST regions that should be replaced by holes.
- Depth-proportional clause weights bias the optimizer toward leaf-level edits rather than coarse structural deletions.
- The approach relaxes strict syntactic minimality in exchange for more localized repairs in practice.
- MAX-SMT is central to the largest consistent subtree stage before model-based completion.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[mora-2024-synthetic-2406-03636]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[mora-2024-synthetic-2406-03636]].
