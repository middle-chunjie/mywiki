---
type: concept
title: Technology Distribution Forecasting
slug: technology-distribution-forecasting
date: 2026-04-20
updated: 2026-04-20
aliases: [技术分布预测]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Technology Distribution Forecasting** (技术分布预测) — predicting a full distribution of company emphasis over technologies for a future time step.

## Key Points

- The paper formalizes the target as the vector `r_i^T = [r_{i,1}^T, ..., r_{i,N}^T]`, where each component is a next-year technology share.
- Company-technology affinity is predicted with `r_hat_uv = sigma(u . v)` after dynamic embeddings are learned for both sides.
- Training uses a BPR-style pairwise objective over triples `(i, j+, j-)` that encode stronger versus weaker technology emphasis.
- The formulation turns firm-level technology planning into a ranking problem over `662` CPC-group technologies.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[wu-2019-deep]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[wu-2019-deep]].
