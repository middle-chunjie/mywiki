---
type: concept
title: InfoNCE Loss
slug: infonce-loss
date: 2026-04-20
updated: 2026-04-20
aliases: [InfoNCE, 信息噪声对比估计损失]
tags: [contrastive-learning, objective-function]
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**InfoNCE Loss** (信息噪声对比估计损失) — a contrastive objective that raises similarity for positive pairs while lowering similarity to negatives, often implemented through a normalized exponential classification form.

## Key Points

- MA-GCL uses InfoNCE as the training objective on top of graph representations produced by two augmented views.
- The paper writes an equivalent squared-distance form over positive and negative node pairs.
- In the MA-GCL implementation, the contrastive loss is applied after a `2`-layer embedding projector.
- The method relies on InfoNCE to extract information shared between views while filtering view-specific noise.
- The theoretical analysis of the asymmetric strategy is motivated by minimizing the positive-pair distance term in the contrastive objective.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[gong-2022-magcl-2212-07035]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[gong-2022-magcl-2212-07035]].
