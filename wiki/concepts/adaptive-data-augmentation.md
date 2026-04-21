---
type: concept
title: Adaptive Data Augmentation
slug: adaptive-data-augmentation
date: 2026-04-20
updated: 2026-04-20
aliases: [自适应数据增强, adaptive graph augmentation]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Adaptive Data Augmentation** (自适应数据增强) — an augmentation strategy that learns how strongly to perturb each training instance or graph connection based on the data itself rather than applying fixed random corruption.

## Key Points

- DCCF replaces random edge or node dropout with a learnable interaction mask over observed user-item edges.
- Each mask value is computed from cosine similarity between disentangled user and item embeddings, then linearly mapped to `[0, 1]`.
- The method constructs both local and global adaptive views, giving the contrastive objective multiple learned perturbations.
- The authors motivate adaptive augmentation as a way to suppress noisy self-supervised signals from misclicks and popularity bias.
- Ablation results show that removing local or disentangled relation masking reduces recommendation quality on all datasets.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[ren-2023-disentangled]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[ren-2023-disentangled]].
