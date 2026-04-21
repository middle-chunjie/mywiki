---
type: concept
title: Process Supervision
slug: process-supervision
date: 2026-04-20
updated: 2026-04-20
aliases: [step-level supervision, 过程监督]
tags: [llm, reasoning, alignment]
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Process Supervision** (过程监督) — a supervision scheme that evaluates intermediate reasoning steps rather than only the final outcome of a model's solution.

## Key Points

- The paper trains a process-supervised reward model by predicting whether each reasoning step is `positive`, `negative`, or `neutral`.
- For incorrect solutions, supervision intentionally stops at the first incorrect step so process supervision differs from outcome supervision only by revealing error location.
- The solution-level PRM score is computed as the product of step-level correctness probabilities, making process quality directly affect ranking.
- On MATH best-of-`1860` search, process supervision outperforms outcome supervision by a substantial margin (`78.2%` vs `72.4%`).
- The paper argues process supervision improves credit assignment and has alignment advantages because it rewards human-endorsed reasoning trajectories directly.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[lightman-2023-lets-2305-20050]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[lightman-2023-lets-2305-20050]].
