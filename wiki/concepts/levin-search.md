---
type: concept
title: Levin Search
slug: levin-search
date: 2026-04-20
updated: 2026-04-20
aliases: [Levin search]
tags: []
source_count: 1
confidence: low
domain_volatility: low
last_reviewed: 2026-04-20
---

## Definition

**Levin search** — a search strategy that prioritizes candidates by a computation-cost-to-probability ratio so that likely and cheap solutions are explored first.

## Key Points

- [[orseau-2018-singleagent-1811-10928]] adapts Levin search from program search to tree search by defining node cost as `d_0(n) / pi(n)`.
- This adaptation yields LevinTS, which expands the fringe node with minimum cost under a policy over action sequences.
- The paper proves a strict upper bound `min_{n in N^g} d_0(n) / pi(n)` on expansions before the first goal is found.
- The analysis explains why pure probability ordering is insufficient: high-probability infinite chains can block low-probability solutions unless depth is priced in.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[orseau-2018-singleagent-1811-10928]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[orseau-2018-singleagent-1811-10928]].
