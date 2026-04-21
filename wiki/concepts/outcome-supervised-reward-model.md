---
type: concept
title: Outcome-Supervised Reward Model
slug: outcome-supervised-reward-model
date: 2026-04-20
updated: 2026-04-20
aliases: [ORM, outcome reward model]
tags: [agents, reward-modeling]
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Outcome-Supervised Reward Model** (结果监督奖励模型) — a reward model trained from final success or failure labels rather than fine-grained stepwise supervision, and used to estimate the probability that a candidate solution is correct.

## Key Points

- The paper trains verifiers as ORMs over task context, trajectory content, and candidate code changes.
- For OpenHands, the verifier predicts `<YES>` or `<NO>` for entire trajectories and converts token log-probabilities into a scalar success score.
- The default verifier dataset mixes off-policy and on-policy trajectories, balancing successes and failures.
- The trained ORM is used to rerank sampled trajectories for inference-time scaling on SWE-Bench.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[pan-2024-training-2412-21139]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[pan-2024-training-2412-21139]].
