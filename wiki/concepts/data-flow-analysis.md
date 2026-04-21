---
type: concept
title: Data-Flow Analysis
slug: data-flow-analysis
date: 2026-04-20
updated: 2026-04-20
aliases: [def-use analysis, 数据流分析]
tags: []
source_count: 1
confidence: low
domain_volatility: low
last_reviewed: 2026-04-20
---

## Definition

**Data-Flow Analysis** (数据流分析) — a static analysis technique that reasons about how definitions, uses, and dependencies of values propagate through program structure.

## Key Points

- [[hooda-2024-do-2402-05980]] uses data-flow analysis to find independent statement blocks for the Independent Swap mutation.
- The same general machinery supports identifying def-use chains for the Def-Use Break mutation.
- The paper relies on data-flow analysis to preserve specificity, ensuring each mutation changes only the targeted predicate as much as possible.
- Predicate-specific mutations depend on analysis quality because false dependencies would invalidate the counterfactual assumption.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[hooda-2024-do-2402-05980]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[hooda-2024-do-2402-05980]].
