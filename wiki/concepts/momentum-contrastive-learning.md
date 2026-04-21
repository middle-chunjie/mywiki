---
type: concept
title: Momentum Contrastive Learning
slug: momentum-contrastive-learning
date: 2026-04-20
updated: 2026-04-20
aliases: [MoCo, 动量对比学习]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Momentum Contrastive Learning** (动量对比学习) — a contrastive learning strategy that uses slowly updated momentum encoders and a queue of past representations to provide large, consistent sets of negative examples.

## Key Points

- CoCoSoDa adopts a MoCo-style queue so each sample is contrasted against `K = 4096` negatives rather than only the current minibatch.
- The momentum code and query encoders are updated by exponential interpolation, with `m = 0.999` in the reported experiments.
- The paper argues that this mechanism gives more consistent negatives than a fixed memory bank and scales better than enlarging the minibatch.
- Larger negative sets are a core reason the method improves sequence-level code and query representations for retrieval.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[shi-2023-cocosoda-2204-03293]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[shi-2023-cocosoda-2204-03293]].
