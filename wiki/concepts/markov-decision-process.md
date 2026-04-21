---
type: concept
title: Markov Decision Process
slug: markov-decision-process
date: 2026-04-20
updated: 2026-04-20
aliases: [MDP, Markovian decision process]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Markov Decision Process** (马尔可夫决策过程) — a sequential decision framework in which the next state depends only on the current state and action, not the full prior history.

## Key Points

- IterResearch formulates long-horizon research as an extended MDP `⟨S, D, E, T⟩` rather than an unstructured prompt loop.
- The state is reconstructed as `s_t = (q, M_t, {a_(t-1), TR_(t-1)})`, making the current report and latest interaction sufficient for the next decision.
- The decision tuple `d_t = (Think_t, M_(t+1), a_t)` jointly updates memory and action selection inside the policy.
- The paper uses the MDP framing to justify geometric reward discounting and stable RL over multi-round trajectories.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[chen-2025-iterresearch-2511-07327]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[chen-2025-iterresearch-2511-07327]].
