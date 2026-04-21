---
type: concept
title: Dataset Construction
slug: dataset-construction
date: 2026-04-20
updated: 2026-04-20
aliases: [dataset curation, 数据集构建]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Dataset Construction** (数据集构建) — the process of sourcing raw material, filtering candidates, defining annotation policy, and validating quality to build a usable benchmark or training corpus.

## Key Points

- CoSQA is built from Bing web queries and Python functions with documentation from CodeSearchNet.
- The pipeline applies heuristic intent filtering, CodeBERT-based candidate retrieval, and agreement-based pruning before release.
- Each retained pair is labeled by at least `3` annotators, and low-agreement examples are removed.
- The authors report both corpus scale and annotation quality statistics, including `20,604` retained labels and average Krippendorff's alpha `0.63`.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[huang-2021-cosqa-2105-13239]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[huang-2021-cosqa-2105-13239]].
