---
type: concept
title: Question Difficulty
slug: question-difficulty
date: 2026-04-20
updated: 2026-04-20
aliases: [prompt difficulty, problem difficulty, 问题难度]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Question Difficulty** (问题难度) — the difficulty of a prompt as measured relative to a base model's ability to solve it, rather than by a human-authored label alone.

## Key Points

- The paper estimates oracle difficulty from pass@1 over `2048` samples per question and groups questions into `5` bins.
- It also defines predicted difficulty by replacing ground-truth correctness with the average verifier score on the same samples.
- Difficulty acts as the sufficient statistic used to choose the compute-optimal test-time strategy.
- Easy bins prefer sequential refinement, medium-hard bins often benefit from search, and the hardest bin sees little gain from either.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[snell-2024-scaling-2408-03314]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[snell-2024-scaling-2408-03314]].
