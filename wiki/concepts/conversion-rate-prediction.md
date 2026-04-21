---
type: concept
title: Conversion Rate Prediction
slug: conversion-rate-prediction
date: 2026-04-20
updated: 2026-04-20
aliases: [CVR Prediction, post-click CVR prediction, 转化率预测]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Conversion Rate Prediction** (转化率预测) — the task of estimating the probability that a clicked impression will lead to a downstream conversion given user, item, and context features.

## Key Points

- This paper formulates post-click CVR as `p(z = 1 | y = 1, x)` under the impression `→` click `→` conversion funnel.
- The paper emphasizes that conversion labels are much sparser than click labels, making deep CVR models strongly data limited.
- CL4CVR improves CVR prediction by augmenting supervised ESMM training with a contrastive objective over unlabeled interaction data.
- Reported gains are substantial in industrial terms: AUC increases of `0.0079` and `0.0066` on two production-style datasets.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[ouyang-2023-contrastive]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[ouyang-2023-contrastive]].
