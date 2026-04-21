---
type: concept
title: Rule-Based Reward
slug: rule-based-reward
date: 2026-04-20
updated: 2026-04-20
aliases: [rule based reward, 规则式奖励]
tags: [reinforcement-learning, optimization]
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Rule-Based Reward** (规则式奖励) — a hand-specified reward signal computed from explicit heuristics or deterministic checks rather than from learned preference models.

## Key Points

- SWE-RL uses a lightweight rule-based reward instead of execution feedback or proprietary reward models.
- Invalid search/replace outputs receive a hard penalty of `-1`, making formatting compliance part of the learning objective.
- Valid outputs are scored by patch similarity between the generated patch and the oracle patch from the merged pull request.
- The reward is continuous in `[0, 1]`, which helps the model learn from partially correct repairs.
- An ablation in the paper shows continuous rewards outperform discrete exact-match rewards on repair quality.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[wei-2025-swerl-2502-18449]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[wei-2025-swerl-2502-18449]].
