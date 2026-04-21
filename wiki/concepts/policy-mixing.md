---
type: concept
title: Policy Mixing
slug: policy-mixing
date: 2026-04-20
updated: 2026-04-20
aliases: [mixed policy, policy mixture]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Policy mixing** — combining multiple policies so search does not rely entirely on a single possibly miscalibrated action distribution.

## Key Points

- [[orseau-2018-singleagent-1811-10928]] analyzes Bayes mixing, where `pi_12 = alpha pi_1 + (1 - alpha) pi_2` over full trajectories.
- It also studies local mixing, where conditional action probabilities are mixed step by step to repair occasional low-probability mistakes.
- The paper's experiments use a simple local mixture with the uniform policy: `tilde(pi)(a|n) = (1 - 0.01) pi(a|n) + 0.01 / 4`.
- On Sokoban, this small amount of noise improves LevinTS's total search effort on hard instances.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[orseau-2018-singleagent-1811-10928]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[orseau-2018-singleagent-1811-10928]].
