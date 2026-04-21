---
type: concept
title: Single-Agent Search
slug: single-agent-search
date: 2026-04-20
updated: 2026-04-20
aliases: [single agent search]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Single-agent search** — search in a non-adversarial environment where one agent seeks any trajectory that reaches a goal state.

## Key Points

- [[orseau-2018-singleagent-1811-10928]] studies deterministic single-agent problems with sparse terminal rewards, where the main goal is simply to find a valid solution.
- The paper represents each node as an action sequence and defines search time as the number of node expansions until any goal node is reached.
- It contrasts this setting with adversarial or stochastic domains where Monte Carlo tree search is often a better fit.
- The paper's guarantees are stated in terms of how a policy allocates probability mass over solution trajectories in this single-agent setting.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[orseau-2018-singleagent-1811-10928]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[orseau-2018-singleagent-1811-10928]].
