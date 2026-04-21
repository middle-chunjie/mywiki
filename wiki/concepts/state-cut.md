---
type: concept
title: State Cut
slug: state-cut
date: 2026-04-20
updated: 2026-04-20
aliases: [state pruning]
tags: []
source_count: 1
confidence: low
domain_volatility: low
last_reviewed: 2026-04-20
---

## Definition

**State cut** — pruning a node because the same state has already been reached by another path with a search cost that is at least as favorable.

## Key Points

- [[orseau-2018-singleagent-1811-10928]] performs a state cut in LevinTS only when the policy is Markovian and the previously expanded node has probability `>=` the current one.
- This rule prevents redundant expansions of duplicate states without breaking the theoretical best-first ordering over states.
- The paper highlights state cuts as a major practical advantage of LevinTS on deterministic domains such as Sokoban.
- A uniform-policy baseline still uses state cuts, which helps explain why it solves some levels despite otherwise weak guidance.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[orseau-2018-singleagent-1811-10928]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[orseau-2018-singleagent-1811-10928]].
