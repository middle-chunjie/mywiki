---
type: concept
title: Unsupervised Contrastive Learning
slug: unsupervised-contrastive-learning
date: 2026-04-20
updated: 2026-04-20
aliases: [无监督对比学习]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Unsupervised Contrastive Learning** (无监督对比学习) — a self-supervised representation learning paradigm that learns embeddings by attracting augmented views of the same instance and repelling different instances without using labels.

## Key Points

- The paper studies unsupervised contrastive learning through the behavior of its softmax-based loss rather than proposing a completely new pretext task.
- It argues that the loss is effective because it is hardness-aware, automatically emphasizing confusing negatives according to their similarity to the anchor.
- Temperature `tau` is treated as the main control variable for how strongly the objective penalizes hard negatives.
- The paper shows that good unsupervised contrastive learning must balance global embedding uniformity with tolerance to semantically similar samples.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[wang-2021-understanding-2012-09740]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[wang-2021-understanding-2012-09740]].
