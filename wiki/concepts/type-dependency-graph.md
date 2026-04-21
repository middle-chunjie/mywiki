---
type: concept
title: Type Dependency Graph
slug: type-dependency-graph
date: 2026-04-20
updated: 2026-04-20
aliases: [TDG, 类型依赖图]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Type Dependency Graph** (类型依赖图) — a directed graph whose nodes represent variable occurrences and typing expressions, and whose edges encode which types can be solved from which others.

## Key Points

- HiTYPER defines a TDG as `` `G = (N, E)` ``, where directed edges indicate type-solvability dependencies between nodes.
- The graph includes symbol, expression, branch, and merge nodes so both data flow and control-flow effects can be modeled.
- Each variable occurrence gets its own node, which lets the system handle Python variables that change type across assignments.
- TDG is the shared substrate that allows forward static inference and backward rejection of neural predictions to interact cleanly.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[peng-2022-static-2105-03595]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[peng-2022-static-2105-03595]].
