---
type: concept
title: Process Reward Model
slug: process-reward-model
date: 2026-04-20
updated: 2026-04-20
aliases: [PRM, process reward model, 过程奖励模型]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Process Reward Model** (过程奖励模型) — a verifier that scores intermediate reasoning steps rather than only the final answer, enabling fine-grained evaluation and search guidance.

## Key Points

- [[dong-2024-progressive-2412-14835]] treats PRM as the core verifier for multimodal reasoning, using step-level scores to rank candidate states during inference.
- AR-MCTS automatically generates PRM supervision from search traces, avoiding manual step annotation.
- The paper first pre-aligns the PRM with step-wise DPO using positive paths with `v_j > 0.8` and negative paths with `v_j = 0`.
- A second point-wise fine-tuning stage teaches the PRM to output calibrated sigmoid scores for individual reasoning states.
- The ablation study shows removing PRM causes one of the largest drops among components, especially on harder multi-step evaluation slices.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[dong-2024-progressive-2412-14835]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[dong-2024-progressive-2412-14835]].
