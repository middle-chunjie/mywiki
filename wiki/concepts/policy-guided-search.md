---
type: concept
title: Policy-Guided Search
slug: policy-guided-search
date: 2026-04-20
updated: 2026-04-20
aliases: [policy guided search]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Policy-guided search** — tree search that uses a policy over actions or trajectories as its guiding signal instead of relying only on heuristic value estimates.

## Key Points

- [[orseau-2018-singleagent-1811-10928]] frames policy-guided search as a special case of heuristic search where the input is a probability distribution `pi` over action sequences.
- Because the guidance is probabilistic, the paper can express node-expansion bounds directly in terms of `pi(n)` for solution nodes.
- The paper introduces both an enumerative variant (LevinTS) and a sampling variant (LubyTS) under this policy-guided view.
- The learned policy comes from A3C, showing how reinforcement learning can provide the guidance while search supplies completeness and stronger guarantees.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[orseau-2018-singleagent-1811-10928]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[orseau-2018-singleagent-1811-10928]].
