---
type: concept
title: Diversity-Driven Sampling
slug: diversity-driven-sampling
date: 2026-04-20
updated: 2026-04-20
aliases: []
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Diversity-Driven Sampling** — a training-time exploration strategy that clusters retrieved candidates and samples across clusters to encourage stylistic coverage instead of collapsing onto near-duplicate items.

## Key Points

- [[gou-2023-diversify]] clusters retrieved templates by Jaccard similarity and samples one template from each cluster during RL training.
- The objective is to prevent the retriever from concentrating all probability mass on templates that are too close to the initial template `z_0`.
- At inference time the sampling scheme is replaced by top-scoring retrieval, so the diversity mechanism mainly affects training.
- The ablation `w/o cluster` substantially degrades diversity and overall BLEU, showing that explicit exploration matters.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[gou-2023-diversify]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[gou-2023-diversify]].
