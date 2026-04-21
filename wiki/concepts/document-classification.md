---
type: concept
title: Document Classification
slug: document-classification
date: 2026-04-20
updated: 2026-04-20
aliases: [Document Classification, 文档分类]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Document Classification** (文档分类) — the task of assigning one or more labels to an entire document based on its overall content.

## Key Points

- [[li-2024-chulo-2410-11119]] evaluates document classification on HP, LUN, EURLEX57K, and Inverted EURLEX57K.
- ChuLo compresses full documents into weighted chunk embeddings so a standard Transformer backbone can classify documents without dropping later content.
- The method reaches `0.6440` accuracy on LUN and improves over prior BERT, Longformer, and chunking baselines on three of four document-classification datasets.
- The paper argues that simply exposing the model to more tokens can add noise, whereas keyphrase-weighted chunking keeps more relevant content.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[li-2024-chulo-2410-11119]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[li-2024-chulo-2410-11119]].
