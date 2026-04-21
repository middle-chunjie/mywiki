---
type: concept
title: Soft Data Augmentation
slug: soft-data-augmentation
date: 2026-04-20
updated: 2026-04-20
aliases: [SoDa, 软数据增强]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Soft Data Augmentation** (软数据增强) — lightweight input perturbation that produces semantically related training views through masking or type-based replacement without requiring strict semantics-preserving program transformations.

## Key Points

- CoCoSoDa introduces four SoDa variants: DM, DR, DRST, and DMST.
- For code, one SoDa transform is randomly sampled each iteration and applied to `15%` of relevant tokens; for queries, only dynamic masking is used.
- The paper contrasts SoDa with heavier semantics-preserving code transformations and argues that SoDa is easier to apply across programming languages.
- Removing all SoDa variants lowers average MRR from `0.788` to `0.758`, indicating that augmentation is a material part of the gain.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[shi-2023-cocosoda-2204-03293]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[shi-2023-cocosoda-2204-03293]].
