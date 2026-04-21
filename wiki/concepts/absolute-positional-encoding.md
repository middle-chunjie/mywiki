---
type: concept
title: Absolute Positional Encoding
slug: absolute-positional-encoding
date: 2026-04-20
updated: 2026-04-20
aliases: [Absolute Positional Encoding, absolute position embedding, 绝对位置编码]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Absolute Positional Encoding** (绝对位置编码) — a positional encoding scheme that assigns each sequence index a learned or fixed embedding and injects it directly into the token representation for attention-based models.

## Key Points

- In [[ke-2021-rethinking-2006-15595]], the standard BERT-style formulation adds `p_i` to token embedding `w_i` before self-attention.
- Expanding the first-layer attention score yields four terms: word-to-word, word-to-position, position-to-word, and position-to-position correlations.
- The paper argues the two mixed terms are noisy because semantic content and absolute position are heterogeneous signals.
- TUPE keeps the absolute position signal but moves it into a separate position-only correlation term inside attention rather than summing embeddings at the input.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[ke-2021-rethinking-2006-15595]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[ke-2021-rethinking-2006-15595]].
