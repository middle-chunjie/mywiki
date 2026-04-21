---
type: concept
title: Overlong Reward Shaping
slug: overlong-reward-shaping
date: 2026-04-20
updated: 2026-04-20
aliases: [soft overlong punishment, 超长奖励塑形]
tags: [reinforcement-learning, reward]
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Overlong Reward Shaping** (超长奖励塑形) — a length-aware reward design that softens the penalty on truncated generations so long but potentially valid reasoning traces do not inject excessive reward noise.

## Key Points

- The paper shows that hard penalties on truncated samples can destabilize RL because correct reasoning may be punished solely for exceeding the generation limit.
- DAPO first validates this diagnosis with overlong filtering, which masks truncated samples and improves training stability.
- The final method adds a soft penalty interval controlled by `L_max` and `L_cache`, with zero penalty below `L_max - L_cache`, a linear penalty inside the cache band, and `-1` beyond `L_max`.
- In the ablation table, overlong filtering and soft overlong punishment contribute to gains from `30` to `36` and then to `41` on AIME 2024 avg@32.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[yu-2025-dapo-2503-14476]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[yu-2025-dapo-2503-14476]].
