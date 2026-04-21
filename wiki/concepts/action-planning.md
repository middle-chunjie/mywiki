---
type: concept
title: Action Planning
slug: action-planning
date: 2026-04-20
updated: 2026-04-20
aliases: [planner policy, 动作规划]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Action Planning** (动作规划) — the process of selecting the next sub-goal, tool, and contextual focus for an agentic system at each interaction step.

## Key Points

- In AgentFlow, the planner is the only trainable module and outputs an action conditioned on `(q, K, M^t)`.
- Each action bundles sub-goal formulation, tool selection, and context retrieval from memory.
- The paper treats planner quality as the main bottleneck for reliable multi-turn tool use.
- Flow-GRPO tuning improves planning enough to reduce tool-calling errors and to shift tool usage toward task-appropriate choices.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[li-2026-intheflow-2510-05592]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[li-2026-intheflow-2510-05592]].
