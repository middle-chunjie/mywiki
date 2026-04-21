---
type: concept
title: Monte Carlo Tree Search
slug: monte-carlo-tree-search
date: 2026-04-20
updated: 2026-04-20
aliases: [MCTS, Monte Carlo Tree Search]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Monte Carlo Tree Search** (蒙特卡罗树搜索) — a search algorithm that incrementally expands a decision tree by balancing exploration and exploitation through value estimates and simulated outcomes.

## Key Points

- [[dainese-2024-generating-2405-15383]] adapts MCTS from action planning to long-form code synthesis, where tree nodes are partial programs and edges are code-generation actions.
- GIF-MCTS uses three action types: generate, improve, and fix, making the search aware of both logical refinement and bug repair.
- Node selection uses a modified UCT score `v_i + C * sqrt(ln N_i / (n_{a=a_i} + ε))` with `C = 0.1` and `ε = 1.0` to discourage overusing a single action type from one parent.
- The method appends only `L = 2` lines when expanding a partial program, while evaluating a completed rollout to estimate the branch value.
- Buggy branches are temporarily kept alive with a high placeholder value so the search will attempt up to `f = 3` repair actions before abandoning them.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[dainese-2024-generating-2405-15383]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[dainese-2024-generating-2405-15383]].
