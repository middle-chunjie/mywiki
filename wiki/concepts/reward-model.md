---
type: concept
title: Reward Model
slug: reward-model
date: 2026-04-20
updated: 2026-04-20
aliases: [reward model, 奖励模型]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Reward Model** (奖励模型) — a model that assigns quality scores to candidate outputs or reasoning trajectories so they can be ranked, filtered, or optimized.

## Key Points

- AirRAG uses a process-supervised reward model as one option for selecting the best reasoning trajectory from multiple MCTS rollouts.
- Training data are synthesized by running AirRAG on partial training sets, sampling positive and negative trajectories, and estimating intermediate-state values with Monte Carlo methods.
- The paper reports that the reward-model verifier is more competitive than Jaccard clustering, embedding-based consistency scoring, or simple averaging over candidates.
- A smaller model such as Qwen2.5-14B-Instruct is fine-tuned for this verification role, trading additional training cost for better trajectory selection.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[feng-2025-airrag-2501-10053]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[feng-2025-airrag-2501-10053]].
