---
type: concept
title: Token-Level Reward
slug: token-level-reward
date: 2026-04-20
updated: 2026-04-20
aliases: [per-token reward, 词元级奖励]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Token-Level Reward** (词元级奖励) — a reinforcement-learning supervision signal assigned to specific generated tokens or token spans instead of only to the final sequence output.

## Key Points

- StepSearch assigns per-step rewards to the final token of each search round rather than using only a terminal answer reward.
- The reward is decomposed into information gain and redundancy penalty so intermediate retrieval behavior affects optimization directly.
- Retrieved `<information>` tokens are masked out of the loss, so token-level reward targets only model-generated reasoning and search actions.
- The paper argues token-level reward improves credit assignment for multi-hop search compared with purely global rewards.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[wang-2025-stepsearch-2505-15107]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[wang-2025-stepsearch-2505-15107]].
