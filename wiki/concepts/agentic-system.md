---
type: concept
title: Agentic System
slug: agentic-system
date: 2026-04-20
updated: 2026-04-20
aliases: [agent-based system, 代理式系统]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Agentic System** (代理式系统) — a problem-solving architecture that decomposes work across specialized modules or agents that coordinate through shared state, explicit roles, and tool use over multiple turns.

## Key Points

- AgentFlow instantiates an agentic system with four modules: planner, executor, verifier, and generator.
- The paper argues that this decomposition is more suitable than a monolithic policy for long-horizon reasoning with diverse tools.
- Shared evolving memory makes the intermediate state explicit rather than hiding all reasoning inside a single context window.
- Unlike many prior agentic systems, AgentFlow trains its planner on-policy inside the live interaction loop.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[li-2026-intheflow-2510-05592]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[li-2026-intheflow-2510-05592]].
