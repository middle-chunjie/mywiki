---
type: concept
title: Time-Series Augmentation
slug: time-series-augmentation
date: 2026-04-20
updated: 2026-04-20
aliases: [temporal augmentation, 时间序列增强]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Time-Series Augmentation** (时间序列增强) — generating transformed views of a time series while preserving the temporal semantics needed for downstream modeling.

## Key Points

- TimesURL proposes FTAug, which combines frequency mixing with random cropping rather than adopting vision-style perturbations directly.
- The paper criticizes flipping and permutation because they can destroy trend information and temporal dependencies.
- Random cropping enforces contextual consistency by aligning overlapping subseries across two augmented views.
- Frequency-domain mixing is used to diversify context while avoiding obviously artificial periodicities or noise patterns.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[liu-2023-timesurl]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[liu-2023-timesurl]].
