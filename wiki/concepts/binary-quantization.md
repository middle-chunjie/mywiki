---
type: concept
title: Binary Quantization
slug: binary-quantization
date: 2026-04-20
updated: 2026-04-20
aliases:
  - 二值量化
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Binary Quantization** (二值量化) — a compression scheme that stores embedding vectors in binary form to reduce memory and retrieval cost.

## Key Points

- The paper evaluates binary quantization explicitly rather than treating compression as an afterthought.
- GOR regularization is introduced partly to make the embedding space more uniform and therefore less brittle under quantization.
- With GOR, retrieval quality drops by `1.90` on MTEB and `2.51` on RTEB after binarization, compared with `3.08` and `3.92` without GOR.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[akram-2026-jinaembeddingsvtext-2602-15547]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[akram-2026-jinaembeddingsvtext-2602-15547]].
