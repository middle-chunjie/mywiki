---
type: concept
title: Agentic Reinforcement Learning
slug: agentic-reinforcement-learning
date: 2026-04-20
updated: 2026-04-20
aliases: [Agentic RL, 智能体强化学习]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Agentic Reinforcement Learning** (智能体强化学习) — reinforcement learning for language-based agents that interact with environments through iterative observations and actions rather than single-shot prediction.

## Key Points

- The paper models agentic RL as a history-augmented decision process where the prompt includes task specification, recent interaction history, current observation, and admissible actions.
- D2Skill addresses the sparse-reward and partial-observability challenges of agentic RL by augmenting the prompt with retrieved reusable skills.
- The framework uses paired baseline and skill-injected rollouts under the same policy to derive hindsight signals for optimization.
- Empirical gains are reported on ALFWORLD and WEBSHOP, showing that reusable memory can materially improve long-horizon agent training.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[tu-2026-dynamic-2603-28716]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[tu-2026-dynamic-2603-28716]].
