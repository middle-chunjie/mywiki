---
type: concept
title: Maximum-Entropy Reinforcement Learning
slug: maximum-entropy-reinforcement-learning
date: 2026-04-20
updated: 2026-04-20
aliases: [entropy-regularized reinforcement learning, max-entropy RL, 最大熵强化学习]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Maximum-Entropy Reinforcement Learning** (最大熵强化学习) — a reinforcement learning objective that maximizes expected reward while also encouraging high-entropy policies for exploration and robustness.

## Key Points

- Q-RAG adopts maximum-entropy RL because long-context retrieval requires effective exploration over many candidate chunks.
- The state value is defined as `` `V^pi(s) = E_a[Q^pi(s, a) - alpha log pi(a | s)]` ``, making policy entropy part of the optimization target.
- Training uses a Boltzmann policy over learned `Q` scores, with temperature `alpha` annealed during optimization.
- The ablation study shows that removing the soft-Q formulation consistently hurts support-fact retrieval quality across context lengths.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[sorokin-2026-qrag-2511-07328]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[sorokin-2026-qrag-2511-07328]].
