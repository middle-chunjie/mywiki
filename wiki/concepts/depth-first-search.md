---
type: concept
title: Depth-First Search
slug: depth-first-search
date: 2026-04-20
updated: 2026-04-20
aliases: [DFS, 深度优先搜索]
tags: []
source_count: 1
confidence: low
domain_volatility: low
last_reviewed: 2026-04-20
---

## Definition

**Depth-First Search** (深度优先搜索) — a search strategy that expands one branch of the state space until termination or failure before backtracking to alternative branches.

## Key Points

- LONGPROC uses explicit DFS procedures for Countdown and Travel Planning instead of asking models for only a final answer.
- For Countdown, each DFS state is the current number set, and actions apply one of `[+, -, *, /]` to a chosen pair.
- For Travel Planning, each DFS state is a partial itinerary, and transitions depend on direct-flight availability and schedule compatibility checks.
- The benchmark exposes whether LCLMs can maintain branching, state updates, and backtracking over long traces rather than merely solve a search problem offline.
- The paper reports that many long-output failures in search tasks come from broken state transitions or premature branch dropping.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[ye-2025-longproc-2501-05414]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[ye-2025-longproc-2501-05414]].
