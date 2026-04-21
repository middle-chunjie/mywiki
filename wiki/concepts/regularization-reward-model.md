---
type: concept
title: Regularization Reward Model
slug: regularization-reward-model
date: 2026-04-20
updated: 2026-04-20
aliases: [regularization RM, reward-regularization model, 正则化奖励模型]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Regularization Reward Model** (正则化奖励模型) — a reward model trained to downscore trajectories that exploit a misspecified dense reward, thereby regularizing RL toward behavior that better matches the intended objective.

## Key Points

- DACO introduces a regularization RM `R_r` because the contribution RM can be gamed by superficially high-scoring code patterns.
- Training data for `R_r` are built by contrasting hacked generations from `\pi_{hack}` with safer generations from SFT or GPT-4 traces.
- During RL, `R_r` is mixed with the multitask reward model so intermediate code steps are rewarded for both contribution and non-hacking behavior.
- Ablation results show removing `R_r` hurts helpfulness and increases code error rate, indicating that regularization materially stabilizes learning.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[wu-2024-daco-2403-02528]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[wu-2024-daco-2403-02528]].
