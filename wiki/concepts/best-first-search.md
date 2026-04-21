---
type: concept
title: Best-First Search
slug: best-first-search
date: 2026-04-20
updated: 2026-04-20
aliases: [best first search]
tags: []
source_count: 1
confidence: low
domain_volatility: low
last_reviewed: 2026-04-20
---

## Definition

**Best-first search** — a search strategy that always expands the state or node with the currently lowest evaluation cost.

## Key Points

- [[orseau-2018-singleagent-1811-10928]] formalizes best-first expansion over states using a generic cost function `cost(n)`.
- LevinTS is shown to expand states in best-first order under its cost `d_0(n) / pi(n)`.
- With Markovian policies, the paper proves that state cuts preserve the property that states are first visited at their lowest cost.
- The paper contrasts policy-based best-first ordering with heuristic planners such as GBFS used inside LAMA.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[orseau-2018-singleagent-1811-10928]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[orseau-2018-singleagent-1811-10928]].
