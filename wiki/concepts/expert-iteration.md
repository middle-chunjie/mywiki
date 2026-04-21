---
type: concept
title: Expert Iteration
slug: expert-iteration
date: 2026-04-20
updated: 2026-04-20
aliases: [iterative self-improvement]
tags: [reinforcement-learning, training]
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Expert Iteration** (专家迭代) — an iterative training procedure that improves a model by collecting successful trajectories from its own search or sampling process and retraining on them.

## Key Points

- [[unknown-nd-leanstar]] applies expert iteration after Lean-CoT to collect successful `(state, thought, tactic)` trajectories verified by [[lean]].
- In each iteration the model samples `K = 32` trajectories per problem with temperature `T = 1.0` and a tactic budget `N = 5`.
- The first round produces `32,231` distinct state-thought-tactic pairs and the second produces roughly `19k` more pairs.
- Training on the union of the original synthetic CoT data and successful self-generated proofs raises miniF2F pass@32 from `32.8%` to `34.0%` and then `34.8%`.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[unknown-nd-leanstar]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[unknown-nd-leanstar]].
