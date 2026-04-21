---
type: concept
title: Multi-Turn Interaction
slug: multi-turn-interaction
date: 2026-04-20
updated: 2026-04-20
aliases: [iterative interaction, 多轮交互]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Multi-Turn Interaction** (多轮交互) — an interaction pattern in which an agent alternates actions and observations over several rounds, updating later behavior using earlier feedback.

## Key Points

- [[wang-2024-executable-2402-01030]] uses multi-turn interaction as the core setting for agent-environment behavior rather than treating tasks as one-shot prediction.
- In CodeAct, each round can use execution outputs or error messages from previous code actions as the next observation.
- M3ToolEval explicitly evaluates this setting, allowing up to `10` interaction turns before termination.
- CodeActInstruct selectively keeps trajectories where the model initially fails and later repairs the solution, so the agent learns from iterative correction.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[wang-2024-executable-2402-01030]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[wang-2024-executable-2402-01030]].
