---
type: entity
title: D2Skill
slug: d2skill
date: 2026-04-20
entity_type: tool
aliases: [Dynamic Dual-Granularity Skill Bank, D2Skill-AgenticRL]
tags: []
---

## Description

D2Skill is the dynamic dual-granularity skill-bank framework introduced in [[tu-2026-dynamic-2603-28716]] for agentic RL. It combines task skills, step skills, hindsight utility estimation, and bank management.

## Key Contributions

- Introduces paired baseline and skill-injected rollouts for hindsight skill valuation.
- Combines reflection-driven skill generation with utility-aware retrieval and pruning.
- Improves ALFWORLD and WEBSHOP performance over GRPO while adding only modest training overhead.

## Related Concepts

- [[agentic-reinforcement-learning]]
- [[skill-bank]]
- [[utility-aware-retrieval]]
- [[skill-pruning]]

## Sources

- [[tu-2026-dynamic-2603-28716]]
