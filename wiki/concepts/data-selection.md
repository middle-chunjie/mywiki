---
type: concept
title: Data Selection
slug: data-selection
date: 2026-04-20
updated: 2026-04-20
aliases: [corpus filtering, 数据选择]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Data Selection** (数据选择) — the process of filtering or reweighting pre-training documents so a model trains preferentially on a subset judged to be more useful.

## Key Points

- The paper compares MeCo against a fastText-based data-selection baseline derived from prior DCLM work.
- That baseline selects the top `70%` of documents from a `250B`-token DCLM pool, incurring extra preprocessing cost to score the whole corpus.
- On the main `1.6B` experiment, data selection improves the average score only slightly (`55.7 -> 56.0`), whereas MeCo reaches `56.7` with no extra model-side computation.
- The authors do not claim that metadata conditioning universally dominates data selection; instead, they argue the two are compatible.
- The comparison supports the broader claim that lightweight grouping signals can improve data efficiency without expensive corpus relabeling passes.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[gao-2025-metadata-2501-01956]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[gao-2025-metadata-2501-01956]].
