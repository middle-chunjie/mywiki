---
type: concept
title: Control-Flow Graph
slug: control-flow-graph
date: 2026-04-20
updated: 2026-04-20
aliases: [CFG, control flow graph, 控制流图]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Control-Flow Graph** (控制流图) — a graph over program operations whose edges encode possible execution order through sequencing, branching, looping, and function calls.

## Key Points

- [[long-2022-multiview]] defines CFG nodes as operations rather than operands, covering standard operations, function calls, and returns.
- The paper uses `PosNext` and `NegNext` edges to represent the true and false branches of conditions.
- Loop back-edges are represented by `IterJump`.
- Interprocedural execution is modeled with `CallNext` and `ReturnNext`, in addition to ordinary `Next` edges.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[long-2022-multiview]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[long-2022-multiview]].
