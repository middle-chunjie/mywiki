---
type: concept
title: Synthetic Data Augmentation
slug: synthetic-data-augmentation
date: 2026-04-20
updated: 2026-04-20
aliases: [synthetic optimization data]
tags: [llm, augmentation, code]
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Synthetic Data Augmentation** — the practice of expanding a training set with automatically generated examples that are filtered to preserve task relevance and quality.

## Key Points

- The paper augments human PIE pairs with GPT-3.5-generated optimization examples rather than relying only on human submissions.
- Synthetic pairs are retained only when the optimized code is at least `5x` faster than the source and semantic duplicates are limited.
- The resulting synthetic set contains `1,485` optimization pairs derived from `3,314` unique generated-program groups.
- For GPT-3.5, adding synthetic data improves `Best@8` speedup from `6.74x` to `6.86x` and `%Opt` from `86.71%` to `87.63%`.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[shypula-2024-performanceimproving-2302-07867]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[shypula-2024-performanceimproving-2302-07867]].
