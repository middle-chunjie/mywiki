---
type: concept
title: Process Explanation Model
slug: process-explanation-model
date: 2026-04-20
updated: 2026-04-20
aliases: [PEM, process explanation model, 过程解释模型]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Process Explanation Model** (过程解释模型) — a generative critic that converts step-level reward signals into natural-language explanations that can guide refinement of intermediate reasoning steps.

## Key Points

- [[sun-2025-rearter-2501-07861]] introduces PEM because a scalar [[process-reward-model]] score cannot directly tell the generator how to repair a weak step.
- PEM receives the current state, the candidate reasoning step, and its PRM score, then generates a critique used for step refinement.
- The paper aligns PEM with PRM by labeling an explanation as positive only when the refined step improves the PRM score.
- PEM is optimized with KTO rather than pairwise DPO, because the collected explanation feedback is binary and potentially noisy.
- The ablation study shows that removing PEM or replacing it with raw PRM scores weakens refinement performance.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[sun-2025-rearter-2501-07861]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[sun-2025-rearter-2501-07861]].
