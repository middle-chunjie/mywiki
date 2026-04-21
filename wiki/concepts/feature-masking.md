---
type: concept
title: Feature Masking
slug: feature-masking
date: 2026-04-20
updated: 2026-04-20
aliases: [特征掩码]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Feature Masking** (特征掩码) — a data-augmentation method that creates alternate training views by hiding entire input feature fields rather than perturbing the representation inside each field.

## Key Points

- The paper treats feature masking as the most relevant prior augmentation baseline for contrastive recommendation and CVR learning.
- In feature-rich CVR settings, masking a user, ad, or context field can remove information that is essential for judging conversion likelihood.
- CL4CVR argues that this coarse masking granularity is the main reason RFM and CFM underperform embedding masking.
- On both datasets, the proposed embedding-masking variant yields higher AUC than the feature-masking baselines.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[ouyang-2023-contrastive]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[ouyang-2023-contrastive]].
