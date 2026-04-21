---
type: concept
title: Model-Based Reinforcement Learning
slug: model-based-reinforcement-learning
date: 2026-04-20
updated: 2026-04-20
aliases: [MBRL, model based reinforcement learning]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Model-Based Reinforcement Learning** (基于模型的强化学习) — a reinforcement-learning paradigm that improves decision making by planning or policy optimization with an explicit predictive model of environment dynamics and rewards.

## Key Points

- [[dainese-2024-generating-2405-15383]] treats the world model as executable Python code rather than a learned latent dynamics network.
- The synthesized model predicts next state, reward, and termination jointly through an `Environment.step` interface, then serves as the planner's internal simulator.
- The paper evaluates model quality not only by transition prediction accuracy but also by normalized return when planning with the generated model.
- CWMs are positioned as especially attractive in low-data regimes because a small curated offline dataset can be enough to validate a handoff from language description to planning.
- The approach is competitive on several discrete-control problems, but still trails stronger alternatives on harder continuous-control tasks.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[dainese-2024-generating-2405-15383]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[dainese-2024-generating-2405-15383]].
