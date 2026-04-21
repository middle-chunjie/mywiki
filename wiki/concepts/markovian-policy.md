---
type: concept
title: Markovian Policy
slug: markovian-policy
date: 2026-04-20
updated: 2026-04-20
aliases: [Markov policy, memoryless policy]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Markovian policy** — a policy whose action probabilities depend only on the current state, not on the full action history.

## Key Points

- [[orseau-2018-singleagent-1811-10928]] defines a policy as Markovian when any two nodes reaching the same state induce identical action distributions.
- LevinTS uses this property to justify state cuts: duplicate states reached with lower probability can be discarded safely.
- The best-first ordering proof depends on the fact that continuation probabilities below duplicate states remain identical under a Markovian policy.
- The experimental Sokoban policy is treated as fixed and Markovian once the neural network is trained.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[orseau-2018-singleagent-1811-10928]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[orseau-2018-singleagent-1811-10928]].
