---
type: concept
title: Triplet Ranking Loss
slug: triplet-ranking-loss
date: 2026-04-20
updated: 2026-04-20
aliases: [margin ranking loss, 三元组排序损失]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Triplet Ranking Loss** (三元组排序损失) — a margin-based objective that pushes a positive match closer than a negative match by at least a fixed margin in an embedding space.

## Key Points

- In the paper's background, deep code search is introduced with a triplet loss over code, correct descriptions, and distracting descriptions.
- The objective uses cosine similarity and enforces `` `cos(c, d+) - cos(c, d-) + epsilon` `` to remain below zero after the `max` operation.
- This loss captures retrieval-style relative ordering rather than only binary relevance labels.
- CDCS later switches to binary classification for fine-tuning, which highlights a design gap between ranking-style search and classification-style optimization.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[gu-2018-deep]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[gu-2018-deep]].
