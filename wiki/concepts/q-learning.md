---
type: concept
title: Q-Learning
slug: q-learning
date: 2026-04-20
updated: 2026-04-20
aliases: [Q学习]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Q-Learning** (Q学习) — an off-policy reinforcement learning method that estimates the value of taking an action in a state and following the best future policy thereafter.

## Key Points

- The paper frames program generation as sequential action selection over program prefixes and optimizes action values for appending Brainfuck tokens.
- ALPHAPROG updates the predicted value `Q(s_t, a_t)` toward `r_t + gamma * max_a Q(s_{t+1}, a)` with `gamma = 1`.
- Rewards are derived from compiler outcomes rather than a hand-specified simulator, including compilation success, coverage, and control-flow complexity.
- The method uses Q-values to choose the next token that best improves long-term fuzzing utility.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[li-2022-alphaprog]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[li-2022-alphaprog]].
