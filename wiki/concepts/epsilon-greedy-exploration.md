---
type: concept
title: Epsilon-Greedy Exploration
slug: epsilon-greedy-exploration
date: 2026-04-20
updated: 2026-04-20
aliases: [epsilon-贪心探索, epsilon-greedy, e-greedy]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Epsilon-greedy exploration** (epsilon-贪心探索) — an action-selection strategy that chooses a random action with probability epsilon and the current best-valued action otherwise.

## Key Points

- ALPHAPROG uses epsilon-greedy exploration to balance discovering new token patterns against exploiting the current Q-network.
- The schedule starts at `epsilon = 1` and decays to `0.01` by subtracting `(1 - 0.01) / 100000` after each prediction.
- This schedule makes exploration dominant early in training and lets the policy become increasingly model-driven later.
- The paper states that exploration effectively stops after episode `20,000`.
- Without such exploration, the validity-only reward quickly converges to trivial repetitive programs that no longer improve coverage.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[li-2022-alphaprog]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[li-2022-alphaprog]].
