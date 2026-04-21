---
type: concept
title: Training Dynamics
slug: training-dynamics
date: 2026-04-20
updated: 2026-04-20
aliases: [training behavior over optimization, 训练动态]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Training Dynamics** (训练动态) — the evolution of model representations, behaviors, and task performance over optimization steps rather than only at the final checkpoint.

## Key Points

- Pythia is explicitly constructed to expose training dynamics by releasing `154` checkpoints for each model.
- The suite controls architecture and data ordering so that changes across checkpoints can be interpreted as training-time effects rather than confounds.
- The paper uses this setup to study the emergence of gender bias, memorization, and term-frequency effects over the course of pretraining.
- Early log-spaced checkpoints at steps `{1, 2, 4, ..., 512}` make very early training behavior observable, not just late-stage convergence.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[biderman-2023-pythia-2304-01373]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[biderman-2023-pythia-2304-01373]].
