---
type: concept
title: On-Policy Learning
slug: on-policy-learning
date: 2026-04-20
updated: 2026-04-20
aliases: [on-policy rl, 同策略学习]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**On-Policy Learning** (同策略学习) — a training regime in which updates are computed from trajectories sampled from the current policy rather than from a fixed offline dataset.

## Key Points

- SCoRe uses online rollouts from the learner itself so the revision policy is trained on the same error distribution it will induce at test time.
- The paper argues that this is necessary because offline correction traces do not transfer well when the learner's first-turn responses drift away from the data-collection policy.
- The HumanEval and MATH improvements over Pair-SFT are used as evidence that on-policy training matters in self-correction settings.
- The authors also experiment with mixing in base-model first attempts to broaden state coverage while keeping the main RL loop on-policy.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[kumar-2024-training-2409-12917]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[kumar-2024-training-2409-12917]].
