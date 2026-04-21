---
type: concept
title: Deep Q-Learning
slug: deep-q-learning
date: 2026-04-20
updated: 2026-04-20
aliases: [深度Q学习, DQN]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Deep Q-Learning** (深度Q学习) — a neural approximation of Q-learning that replaces a tabular action-value function with a trainable deep network.

## Key Points

- ALPHAPROG uses deep Q-learning because the state space of partial programs is too large for tabular methods.
- The action-value network is built from an LSTM encoder with `128` neurons and two fully connected layers of sizes `100` and `512`.
- The network outputs `8` scores, one for each Brainfuck token that can be appended next.
- Training combines compiler-derived rewards with epsilon-greedy exploration to avoid collapsing immediately to trivial valid programs.
- The architecture allows the generator to bootstrap from scratch without a seed corpus of valid programs.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[li-2022-alphaprog]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[li-2022-alphaprog]].
