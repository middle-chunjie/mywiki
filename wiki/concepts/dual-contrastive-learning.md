---
type: concept
title: Dual Contrastive Learning
slug: dual-contrastive-learning
date: 2026-04-20
updated: 2026-04-20
aliases: [双重对比学习, DCL]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Dual Contrastive Learning** (双重对比学习) — a contrastive objective that simultaneously aligns user representations with targets at two semantic levels, here item level and category level.

## Key Points

- HPM defines separate losses `L_cl_item` and `L_cl_cate` to supervise low-level item preference and high-level category preference.
- Positive samples are the semantics-enhanced target item and target category embeddings, while negatives are other targets in the same batch.
- Unlike augmentation-heavy sequential contrastive methods, the paper avoids perturbing the original sequence order with masking or reordering.
- Removing DCL degrades HR@5 and NDCG@5 in the ablation study, showing that dual-level supervision matters beyond the ranking loss alone.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[huang-2023-dual]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[huang-2023-dual]].
