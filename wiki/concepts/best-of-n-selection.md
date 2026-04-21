---
type: concept
title: Best-of-N Selection
slug: best-of-n-selection
date: 2026-04-20
updated: 2026-04-20
aliases: [best of N selection, BoN selection, 最优采样重排]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Best-of-N Selection** (最优采样重排) — an inference strategy that samples multiple candidate solutions and chooses the highest-scoring one under a verifier or reward model.

## Key Points

- [[wang-2024-mathshepherd-2312-08935]] evaluates verifiers by sampling `256` candidate math solutions for each problem and reranking them.
- The paper uses best-of-`N` as the main verification protocol for comparing self-consistency, ORM, and PRM.
- For PRM, the score of a full solution is taken as the minimum score among its steps, so one bad step can sink the trajectory.
- The authors also combine reward-model reranking with answer-group aggregation under self-consistency.
- Performance gains become larger as the number of candidates increases, especially on the harder MATH benchmark.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[wang-2024-mathshepherd-2312-08935]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[wang-2024-mathshepherd-2312-08935]].
