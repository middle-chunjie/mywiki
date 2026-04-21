---
type: concept
title: Transition Function
slug: transition-function
date: 2026-04-20
updated: 2026-04-20
aliases: [State transition function, 状态转移函数]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Transition Function** (状态转移函数) — a mapping from the current state and action to the next state, encoding the environment dynamics assumed by a world model or MDP.

## Key Points

- WorldCoder represents the learned dynamics explicitly as Python code implementing `\hat{T}: S \times A \to S`.
- The paper assumes deterministic environments, so each `(state, action)` pair maps to exactly one predicted next state.
- The transition function is separated from the reward logic to improve modularity and transfer across new goals.
- Data consistency `\phi_1` requires the synthesized `\hat{T}` to replay every observed state transition in the interaction dataset.
- Transfer across related environments works by refining previously learned transition code with new counterexamples rather than rebuilding from scratch.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[tang-2024-worldcoder-2402-12275]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[tang-2024-worldcoder-2402-12275]].
