---
type: concept
title: Matryoshka Representation Learning
slug: matryoshka-representation-learning
date: 2026-04-20
updated: 2026-04-20
aliases:
  - MRL
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Matryoshka Representation Learning** — an embedding-training scheme that arranges useful information so truncated prefixes of the vector remain effective at downstream tasks.

## Key Points

- The v5 text models are trained so their embeddings can be shortened after training for lower storage and compute cost.
- The paper evaluates progressively smaller embedding dimensions and finds the retrieval degradation becomes pronounced below `256` dimensions.
- This truncation behavior is positioned as a practical efficiency feature alongside long-context support and quantization robustness.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[akram-2026-jinaembeddingsvtext-2602-15547]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[akram-2026-jinaembeddingsvtext-2602-15547]].
