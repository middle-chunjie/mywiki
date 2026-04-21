---
type: concept
title: Universal Representation Learning
slug: universal-representation-learning
date: 2026-04-20
updated: 2026-04-20
aliases: [通用表征学习]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Universal Representation Learning** (通用表征学习) — learning a single representation space that remains useful across heterogeneous downstream tasks without task-specific pretraining of the encoder.

## Key Points

- TimesURL explicitly optimizes for one encoder to support forecasting, imputation, classification, anomaly detection, and transfer learning.
- The paper argues universality requires both segment-level and instance-level information rather than only one semantic granularity.
- Positive-pair construction and negative-pair hardness are treated as core determinants of whether representations transfer across tasks.
- The reported evaluation spans six task families to operationalize "universal" as cross-task rather than single-benchmark performance.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[liu-2023-timesurl]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[liu-2023-timesurl]].
