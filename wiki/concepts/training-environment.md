---
type: concept
title: Training Environment
slug: training-environment
date: 2026-04-20
updated: 2026-04-20
aliases: [learning environment, agent training environment]
tags: [agents, training]
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Training Environment** (训练环境) — a structured environment that provides tasks, observations, actions, and reward signals so an agent can improve through interaction rather than static supervision alone.

## Key Points

- The paper frames SWE-Gym as the missing training environment for software engineering agents.
- Each task instance includes repository context, a runnable environment, and test-based feedback.
- The environment supports both policy learning from successful trajectories and verifier training from success/failure labels.
- The authors argue that lack of such environments is the main reason SWE agents had relied on prompting and proprietary models.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[pan-2024-training-2412-21139]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[pan-2024-training-2412-21139]].
