---
type: concept
title: Category-specific awareness head
slug: category-specific-awareness-head
date: 2026-04-20
updated: 2026-04-20
aliases: [CAH]
tags: [feature-adaptation, retrieval]
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Category-specific awareness head** — a lightweight feature adaptation head that removes species-level discrepancies from frozen backbone features while preserving subcategory-sensitive evidence.

## Key Points

- CAH is the feature-adaptation component in FRPT and operates on the last convolutional feature map.
- It combines original features with instance-normalized features through channel attention rather than replacing them outright.
- The head is supervised by subcategory labels so that the resulting representation focuses on within-meta-category distinctions.
- Ablations show that CAH improves Recall@1 strongly even without DPP and performs best when combined with DPP.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[wang-2023-finegrained-2207-14465]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[wang-2023-finegrained-2207-14465]].
