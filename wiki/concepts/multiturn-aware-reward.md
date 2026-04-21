---
type: concept
title: Multiturn-aware Reward
slug: multiturn-aware-reward
date: 2026-04-20
updated: 2026-04-20
aliases: [MR, multi-turn-aware reward, 多轮感知奖励]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Multiturn-aware Reward** (多轮感知奖励) — a reward for a model response defined by its expected effect on future conversation trajectories and final collaboration outcomes, not only its immediate local quality.

## Key Points

- The paper defines MR as an expectation over sampled future turns conditioned on the current history and candidate response.
- MR is designed to align training with conversation-level success `R*(t | g)` instead of the sum of single-turn rewards.
- The reward can combine extrinsic task metrics with intrinsic efficiency and interactivity terms.
- MR is used both for online RL updates and for generating synthetic preference or supervision data.
- Ablations show that MR with nonzero rollout windows outperforms immediate-reward variants that ignore future effects.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[wu-2025-collabllm-2502-00640]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[wu-2025-collabllm-2502-00640]].
