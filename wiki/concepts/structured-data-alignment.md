---
type: concept
title: Structured Data Alignment
slug: structured-data-alignment
date: 2026-04-20
updated: 2026-04-20
aliases: [SDA, 结构化数据对齐]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Structured Data Alignment** (结构化数据对齐) — a contrastive pretraining objective that aligns structured documents with semantically matched natural-language passages in a shared embedding space.

## Key Points

- In SANTA, positive pairs are code-documentation pairs and product-description/bullet-point pairs that naturally share semantics.
- The loss `L_SDA` is a contrastive objective over matched structured-unstructured pairs with in-batch negatives.
- SDA is the main source of performance gain in the ablation study, especially for zero-shot code retrieval.
- The objective is intended to bridge the modality gap between structured and unstructured text while preserving retrieval discrimination.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[li-2023-structureaware-2305-19912]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[li-2023-structureaware-2305-19912]].
