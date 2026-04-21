---
type: concept
title: In-Batch Negative Sampling
slug: in-batch-negative-sampling
date: 2026-04-20
updated: 2026-04-20
aliases: [Batch Negative Sampling, 批内负例采样]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**In-Batch Negative Sampling** (批内负例采样) — a training strategy that reuses other examples in the same mini-batch as negative candidates for each positive pair in a contrastive objective.

## Key Points

- The paper adopts in-batch negatives together with random negatives as the simplest possible dense-retrieval training recipe.
- This choice is deliberate: the authors want to minimize confounding effects from sophisticated sampling tricks while studying scaling behavior.
- Each model is fine-tuned for `10,000` steps and uses `256` negatives per step under a contrastive ranking loss.
- By standardizing the negative-sampling strategy across model sizes and data sizes, the paper makes the fitted scaling exponents easier to interpret.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[fang-2024-scaling-2403-18684]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[fang-2024-scaling-2403-18684]].
