---
type: concept
title: Generalizability
slug: generalizability
date: 2026-04-20
updated: 2026-04-20
aliases: [泛化能力, generalization ability]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Generalizability** (泛化能力) — the ability of a model to maintain predictive quality beyond the specific examples seen during training, especially when data are limited.

## Key Points

- The paper treats generalizability as a first-class requirement because chemical space is vast while labeled molecular datasets are comparatively small.
- MGCN is designed to improve generalization through rotation and translation invariance derived from distance-based inputs rather than raw coordinates.
- The model also avoids dependence on atom ordering by using element-wise interaction operations and per-atom processing.
- In the reduced-data study, MGCN retains the best MAE at `50k`, `100k`, and full-data training sizes on QM9, supporting the paper's generalization claim.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[lu-2019-molecular]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[lu-2019-molecular]].
