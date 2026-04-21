---
type: concept
title: Data-Flow Graph
slug: data-flow-graph
date: 2026-04-20
updated: 2026-04-20
aliases: [DFG, data flow graph, 数据流图]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Data-Flow Graph** (数据流图) — a program graph whose nodes denote operands and whose edges encode how values move through operations, calls, and returns.

## Key Points

- [[long-2022-multiview]] uses the DFG to emphasize semantic value dependencies rather than syntax.
- The paper distinguishes non-temporary operands appearing in source code from temporary operands created during execution.
- DFG edges include operation edges for operators such as `=`, `+`, `-`, `*`, `/`, `>`, `<`, and `==`.
- Function-related data movement is modeled with `Argument` and `ReturnTo` edges.
- In the ablation study, removing the DFG causes the largest drop among all view removals, indicating it is the most critical view in MVG.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[long-2022-multiview]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[long-2022-multiview]].
