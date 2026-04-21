---
type: concept
title: Outcome Reward Model
slug: outcome-reward-model
date: 2026-04-20
updated: 2026-04-20
aliases: [ORM, outcome reward model, 结果奖励模型]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Outcome Reward Model** (结果奖励模型) — a verifier that judges the quality of an entire reasoning trajectory based mainly on the final outcome instead of intermediate steps.

## Key Points

- [[dong-2024-progressive-2412-14835]] positions ORM as a stronger baseline than simple self-correction or self-consistency, but still too coarse for hard multi-step multimodal reasoning.
- The paper argues ORM feedback is sparse because it only evaluates completed reasoning paths.
- AR-MCTS outperforms ORM most clearly on We-Math S3, where step-level errors matter more and PRM can supervise intermediate reasoning quality.
- In the authors' setup, ORM is trained on the same sampled reasoning data as PRM, except labels come directly from final correctness rather than step-wise AR-MCTS annotations.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[dong-2024-progressive-2412-14835]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[dong-2024-progressive-2412-14835]].
