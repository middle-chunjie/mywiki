---
type: concept
title: Thresholded Mean Average Precision
slug: thresholded-mean-average-precision
date: 2026-04-20
updated: 2026-04-20
aliases: [t-mAP, 阈值化平均精度均值]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Thresholded Mean Average Precision** (阈值化平均精度均值) — a retrieval metric that averages mean average precision after suppressing predictions at thresholds chosen according to false positive rates, thereby measuring both ranking quality and score margin.

## Key Points

- [[wu-2023-forb-2309-16249]] defines `t-mAP = (1 / τ(1)) ∫_0^{τ(1)} mAP(t) dt`, where `τ(x)` is the score threshold producing false positive rate `1 - x`.
- In practice, the paper approximates the integral with `11` thresholds: `0, τ(0.1), ..., τ(1.0)`.
- The metric is motivated by the claim that standard `mAP` does not sufficiently penalize confident false positives on OOD-style queries.
- On FORB, `t-mAP` reveals stronger separation ability for methods such as `FIRe` and even `BoW` than would be obvious from `mAP` alone.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[wu-2023-forb-2309-16249]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[wu-2023-forb-2309-16249]].
