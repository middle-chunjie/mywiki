---
type: concept
title: Imitation Learning
slug: imitation-learning
date: 2026-04-20
updated: 2026-04-20
aliases: [behavior cloning, 模仿学习]
tags: [agents, llm, training]
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Imitation Learning** (模仿学习) — a learning paradigm in which a model is trained to reproduce expert trajectories or actions from supervised demonstrations.

## Key Points

- In [[gou-2024-tora-2309-17452]], imitation learning is the first training stage and uses tool-integrated reasoning trajectories synthesized by GPT-4.
- The optimization target maximizes the likelihood of subsequent rationale-and-program steps conditioned on the question and prior trajectory context.
- The supervision data comes from TORA-CORPUS, a filtered collection of about `16k` valid trajectories over GSM8k and MATH.
- The paper argues that imitation alone is strong but still too restrictive, motivating later [[output-space-shaping]].

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[gou-2024-tora-2309-17452]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[gou-2024-tora-2309-17452]].
