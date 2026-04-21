---
type: concept
title: Reward Shaping
slug: reward-shaping
date: 2026-04-20
updated: 2026-04-20
aliases: [shaped reward]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Reward Shaping** (奖励塑形) — modifying the training reward signal so that learning is guided not only by final success but also by preferences over how success is achieved.

## Key Points

- IterResearch uses geometric discounting as a reward-shaping mechanism to prefer efficient successful trajectories.
- The shaped reward still depends on the same terminal correctness label `R_T ∈ {0, 1}` rather than manually scoring intermediate actions.
- The paper motivates shaping through deployment cost and latency: longer research trajectories are more expensive even if they answer correctly.
- Theoretical analysis in the appendix illustrates that identical early steps in shorter trajectories receive systematically larger rewards.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[chen-2025-iterresearch-2511-07327]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[chen-2025-iterresearch-2511-07327]].
