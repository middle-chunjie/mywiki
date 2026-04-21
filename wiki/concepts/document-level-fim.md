---
type: concept
title: Document-Level FIM
slug: document-level-fim
date: 2026-04-20
updated: 2026-04-20
aliases: [document level FIM, 文档级FIM]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Document-Level FIM** (文档级FIM) — a fill-in-the-middle augmentation strategy that transforms whole documents before they are packed and chunked into model contexts.

## Key Points

- The document is split into prefix, middle, and suffix before tokenization or packing, then serialized in FIM order with sentinel tokens.
- After later chunking, the resulting contexts may omit part of the prefix or suffix, creating fragmented training examples.
- The paper finds that this broken-data effect hurts infilling quality more than it hurts ordinary left-to-right evaluation.
- Despite the performance gap, the authors note document-level FIM may still be attractive when implementation simplicity is the primary constraint.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[bavarian-2022-efficient-2207-14255]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[bavarian-2022-efficient-2207-14255]].
