---
type: concept
title: Tree Search
slug: tree-search
date: 2026-04-20
updated: 2026-04-20
aliases: [search tree, 树搜索]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Tree Search** (树搜索) — a planning strategy that explicitly expands and evaluates branching candidate states or action sequences in a tree rather than following a single greedy trajectory.

## Key Points

- ToolTree reframes tool use as search over executable trajectories whose nodes represent agent states and whose edges represent tool actions.
- The paper contrasts tree search against greedy controllers such as zero-shot, CoT, and ReAct, which do not systematically explore alternatives.
- Search quality is improved by combining predictive scores before execution with grounded rewards after execution.
- The work evaluates several search baselines, including Best-First, ToT, A*, and LATS, and reports that ToolTree performs best under matched budgets.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[yang-2026-tooltree-2603-12740]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[yang-2026-tooltree-2603-12740]].
