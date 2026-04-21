---
type: concept
title: Model Augmentation
slug: model-augmentation
date: 2026-04-20
updated: 2026-04-20
aliases: [模型增强]
tags: [self-supervised-learning, representation-learning]
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Model Augmentation** (模型增强) — an augmentation strategy that perturbs architectural structure or computation pathways of a model, rather than only perturbing raw inputs, to create alternative training views.

## Key Points

- MA-GCL introduces model augmentation as the central paradigm shift of the paper.
- The paper instantiates model augmentation with asymmetric propagation depth, random propagation depth, and shuffled operator order.
- These perturbations are applied to GNN view encoders while keeping learnable parameters shared across views.
- The design goal is to make contrastive views neither too close nor too far.
- The method is presented as compatible with conventional graph data augmentation rather than a replacement for it.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[gong-2022-magcl-2212-07035]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[gong-2022-magcl-2212-07035]].
