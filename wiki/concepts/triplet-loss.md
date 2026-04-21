---
type: concept
title: Triplet Loss
slug: triplet-loss
date: 2026-04-20
updated: 2026-04-20
aliases: [三元组损失]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Triplet Loss** (三元组损失) — a margin-based ranking objective that pulls an anchor closer to a positive example than to a negative example in embedding space.

## Key Points

- XLIR trains on triplets `<b, s+, s->` consisting of a binary sample, its paired source sample, and a randomly selected negative source sample.
- The paper optimizes `` `max(0, alpha - sim(b, s+) + sim(b, s-))` `` with margin `` `alpha = 0.06` ``.
- Cosine similarity is used for both positive and negative pair scoring.
- This objective is the main mechanism that aligns source and binary embeddings after IR encoding.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[gui-2022-crosslanguage-2201-07420]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[gui-2022-crosslanguage-2201-07420]].
