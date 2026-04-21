---
type: concept
title: Image-Text Classification
slug: image-text-classification
date: 2026-04-20
updated: 2026-04-20
aliases: [multimodal image-text classification, vision-language classification, 图文分类]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Image-Text Classification** (图文分类) — a multimodal classification task in which both an image and one or more text fields (caption, headline, abstract, recipe, etc.) are available as input and must be jointly used to assign a category label.

## Key Points

- UPMC-Food-101 (101 food categories, ~87K samples with recipe text + images) and N24News (24 news categories, ~61K samples with headline/caption/abstract + images) are the two standard benchmarks in this line of work.
- Fusion strategies fall into aggregation-based (early/late/hybrid concatenation) and alignment-based (cross-modal contrastive or attention-based) approaches; state-of-the-art methods typically combine both.
- Unimodal accuracy varies substantially across text sources within the same dataset: abstract-only reaches 78–80% on N24News while headline-only reaches only 72%.
- The choice of text encoder (BERT vs. RoBERTa) significantly impacts multimodal results, with RoBERTa-based models generally outperforming BERT-based ones on N24News.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[zou-2023-unismmc-2305-09299]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[zou-2023-unismmc-2305-09299]].
