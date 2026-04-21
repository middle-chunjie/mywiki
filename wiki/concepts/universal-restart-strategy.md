---
type: concept
title: Universal Restart Strategy
slug: universal-restart-strategy
date: 2026-04-20
updated: 2026-04-20
aliases: [restart schedule, universal restarting strategy]
tags: []
source_count: 1
confidence: low
domain_volatility: low
last_reviewed: 2026-04-20
---

## Definition

**Universal restart strategy** — a schedule for repeatedly restarting a randomized procedure with varying budgets when the unknown halting time distribution is not known in advance.

## Key Points

- [[orseau-2018-singleagent-1811-10928]] imports restart theory into tree search to handle unknown solution depths.
- LubyTS samples trajectories whose allowed depths follow the `A6519` sequence rather than a fixed cutoff.
- The paper proves an expectation bound `min_d d + (d / pi^+_d) * (log_2(d / pi^+_d) + 6.1)` for the resulting search process.
- The restart view explains why LubyTS is attractive when many solution nodes share large cumulative probability.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[orseau-2018-singleagent-1811-10928]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[orseau-2018-singleagent-1811-10928]].
