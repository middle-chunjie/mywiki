---
type: concept
title: Optimism under Uncertainty
slug: optimism-under-uncertainty
date: 2026-04-20
updated: 2026-04-20
aliases: [Optimistic exploration, 不确定性下的乐观原则]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Optimism under Uncertainty** (不确定性下的乐观原则) — a decision-making principle that prefers models or actions consistent with current evidence while still predicting an achievable positive outcome in uncertain parts of the state space.

## Key Points

- WorldCoder formalizes optimism as a logical constraint `\phi_2` requiring that the learned program admit some action sequence from the initial state to positive reward.
- The optimism constraint is paired with a strict data-fit constraint `\phi_1`, so the agent must both explain observed transitions and imagine a plausible rewarding plan.
- In sparse-reward settings, this objective induces goal-directed exploration instead of unguided random probing.
- The paper argues that optimism enables zero-shot adaptation to new natural-language goals by forcing the reward model to cover previously unseen contexts.
- Appendix A reports that, in MiniGrid UnlockPickup, adding optimism lets the agent learn the correct world model within `100` actions whereas PPO fails even after `3 × 10^8` actions.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[tang-2024-worldcoder-2402-12275]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[tang-2024-worldcoder-2402-12275]].
