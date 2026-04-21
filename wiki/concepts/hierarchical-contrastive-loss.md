---
type: concept
title: Hierarchical Contrastive Loss
slug: hierarchical-contrastive-loss
date: 2026-04-20
updated: 2026-04-20
aliases: [层次化对比损失]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Hierarchical Contrastive Loss** (层次化对比损失) — a contrastive objective applied across multiple temporal resolutions so representations preserve information at more than one scale.

## Key Points

- TimesURL applies temporal-wise and instance-wise contrastive losses after hierarchical max pooling along the time axis.
- The paper uses the multi-scale setup to capture both timestamp-local structure and broader sample-level semantics.
- Double Universums are injected as additional hard negatives inside both branches of the hierarchical objective.
- The authors note that upper pooling levels alone lose important temporal variation, motivating the joint use of reconstruction.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[liu-2023-timesurl]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[liu-2023-timesurl]].
