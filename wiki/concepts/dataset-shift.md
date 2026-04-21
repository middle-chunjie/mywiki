---
type: concept
title: Dataset Shift
slug: dataset-shift
date: 2026-04-20
updated: 2026-04-20
aliases: [distribution shift, 数据集偏移]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Dataset Shift** (数据集偏移) — a mismatch between source and target data distributions that changes how a trained model behaves at test time.

## Key Points

- The paper models shift in ODQA through the joint question-context distribution and the conditional answer distribution.
- It organizes target domains into four cases: no shift, [[label-shift]], [[covariate-shift]], and full shift.
- Shift type is estimated by comparing source-model-induced distributions against uniform and gold references rather than training a fully supervised target model.
- The diagnosed shift category predicts which interventions are more likely to help: mild shifts respond better to zero-shot adaptation, while full shift benefits more from few-shot data generation.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[dua-2023-adapt]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[dua-2023-adapt]].
