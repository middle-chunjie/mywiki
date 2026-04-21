---
type: concept
title: Agent Monitoring
slug: agent-monitoring
date: 2026-04-20
updated: 2026-04-20
aliases: [agent oversight, 智能体监控]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Agent Monitoring** (智能体监控) — the inspection of an autonomous agent's actions, tool use, and reasoning traces to judge whether its behavior aligns with user intent and safety requirements.

## Key Points

- The paper studies monitoring over full agent trajectories rather than single outputs.
- Monitoring targets covert side-task behavior, where an agent can appear helpful while pursuing a harmful objective.
- The work treats long-horizon computer-use and tool-calling agents as the relevant deployment setting.
- Reliability depends strongly on whether the agent knows it is being monitored and on how the monitor parses long trajectories.
- Human oversight is modeled as an escalation stage rather than a full replacement for LLM-based monitoring.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[kale-2026-reliable-2508-19461]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[kale-2026-reliable-2508-19461]].
