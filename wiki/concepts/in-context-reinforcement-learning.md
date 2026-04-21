---
type: concept
title: In-Context Reinforcement Learning
slug: in-context-reinforcement-learning
date: 2026-04-20
updated: 2026-04-20
aliases: [ICRL]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**In-Context Reinforcement Learning** — a reinforcement-learning regime in which auxiliary task guidance is supplied inside rollout context during training, while policy optimization encourages the model to outgrow that dependence.

## Key Points

- Skill0 defines ICRL as training with explicit skill context during rollouts but removing all skills at inference time.
- The policy consumes rendered context that includes both interaction history and selected skill files, so RL updates act on behavior under scaffolded conditions.
- Skill0 augments task reward with a compression bonus `log(c_t)` on successful trajectories, coupling skill use with compact context construction.
- The objective uses grouped rollouts, normalized advantages, clipping, and KL regularization to stabilize policy improvement.
- ICRL is meant to bridge two extremes: pure zero-shot RL without scaffolding and permanent prompt-based skill dependence.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[lu-2026-skill-2604-02268]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[lu-2026-skill-2604-02268]].
