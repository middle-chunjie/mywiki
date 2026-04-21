---
type: concept
title: Temporal-Difference Learning
slug: temporal-difference-learning
date: 2026-04-20
updated: 2026-04-20
aliases: [TD learning, 时序差分学习]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Temporal-Difference Learning** (时序差分学习) — a reinforcement learning method that updates value estimates by bootstrapping from later value predictions instead of waiting for full Monte Carlo returns.

## Key Points

- Q-RAG trains its retriever with a TD-style value-learning objective instead of supervised trajectory imitation.
- The paper adopts PQN as the backbone and replaces one-step targets with `lambda`-returns to improve stability and learning speed.
- Soft state values are computed from the current discrete action set, so TD targets remain compatible with entropy-regularized control.
- Target networks are important in this setting: removing them sharply reduces support-fact retrieval accuracy and greatly increases variance.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[sorokin-2026-qrag-2511-07328]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[sorokin-2026-qrag-2511-07328]].
