---
type: concept
title: Sparse Reward
slug: sparse-reward
date: 2026-04-20
updated: 2026-04-20
aliases: [delayed reward, 稀疏奖励]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Sparse Reward** (稀疏奖励) — a supervision regime in which useful reward signals arrive only at a few points, often only at the end of a multi-step trajectory.

## Key Points

- The paper frames agentic tool use as a sparse-reward problem because success is judged from the final answer rather than from each intermediate tool call.
- Sparse rewards become more difficult as reasoning horizons lengthen and errors compound across turns.
- Flow-GRPO addresses this by broadcasting one verifiable final-outcome reward to every planner decision in a rollout.
- Group-normalized advantages are used to make sparse binary rewards more stable for optimization.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[li-2026-intheflow-2510-05592]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[li-2026-intheflow-2510-05592]].
