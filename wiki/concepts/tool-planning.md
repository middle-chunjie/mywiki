---
type: concept
title: Tool Planning
slug: tool-planning
date: 2026-04-20
updated: 2026-04-20
aliases: [tool orchestration, 工具规划]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Tool Planning** (工具规划) — deciding which external tools to invoke, in what order, and with what arguments so an agent can complete a multi-step task efficiently and correctly.

## Key Points

- This paper formalizes tool planning over a tool library, state space, action space, and reward function instead of treating each tool decision independently.
- ToolTree argues that planning quality depends on evaluating entire executable tool trajectories, not just selecting the locally best next tool.
- The method uses MCTS-style search to allocate budget across alternative tool sequences under a fixed rollout cap.
- Closed-set benchmarks stress typed multi-hop planning over small fixed tool suites, while open-set benchmarks add API retrieval and larger branching factors.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[yang-2026-tooltree-2603-12740]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[yang-2026-tooltree-2603-12740]].
