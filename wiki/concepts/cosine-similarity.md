---
type: concept
title: Cosine Similarity
slug: cosine-similarity
date: 2026-04-20
updated: 2026-04-20
aliases: [余弦相似度, cosine distance]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Cosine Similarity** (余弦相似度) — a similarity measure based on the angle between two vectors, commonly used to compare embedding representations.

## Key Points

- BLANCA ranking tasks sort answers or compare pairs by cosine distance between embeddings.
- The ranking task converts answer popularity into scores and fine-tunes models with cosine similarity loss.
- The hierarchy and usage tasks regress from semantic structure to cosine-distance behavior in embedding space.
- Much of the paper's analysis is phrased as how well cosine distance tracks semantic relatedness in code-related text.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[abdelaziz-2022-can]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[abdelaziz-2022-can]].
