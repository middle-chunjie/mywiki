---
type: concept
title: Multi-Label Text Classification
slug: multi-label-text-classification
date: 2026-04-20
updated: 2026-04-20
aliases: [multi-label classification, 多标签文本分类]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Multi-Label Text Classification** (多标签文本分类) — a classification problem in which each text can be assigned multiple labels simultaneously rather than exactly one mutually exclusive class.

## Key Points

- [[liu-2023-enhancing]] frames HTC as a special case of multi-label classification where the labels additionally follow a hierarchy.
- The output layer flattens the hierarchy into binary label decisions and optimizes binary cross-entropy across all `K` labels.
- The document encoder is trained to support multiple correlated labels through label-aware attention and hierarchy-driven contrastive learning.
- Reported evaluation uses Macro-F1 and Micro-F1 because class imbalance and varying label frequencies make single-label accuracy insufficient.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[liu-2023-enhancing]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[liu-2023-enhancing]].
