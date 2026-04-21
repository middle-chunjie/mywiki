---
type: concept
title: Iterative Refinement
slug: iterative-refinement
date: 2026-04-20
updated: 2026-04-20
aliases: [iterative revision, 迭代式精炼]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Iterative Refinement** (迭代式精炼) — a repeated revision process that challenges draft tasks, removes shortcuts, and preserves only cases that satisfy stricter difficulty constraints.

## Key Points

- Stage 2 of InfoMosaic-Flow uses a web-only verifier to attack draft questions through condition decomposition and fuzzing.
- A task is retained only when web search alone fails and no single condition determines the answer.
- This step is empirically important: removing it inflates benchmark accuracy by admitting shortcut-solvable tasks.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[du-2026-infomosaicbench-2510-02271]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[du-2026-infomosaicbench-2510-02271]].
