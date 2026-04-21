---
type: concept
title: Untied Positional Encoding
slug: untied-positional-encoding
date: 2026-04-20
updated: 2026-04-20
aliases: [TUPE, Transformer with Untied Positional Encoding, 解耦位置编码]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Untied Positional Encoding** (解耦位置编码) — a positional encoding design that computes content correlations and position correlations with separate parameterizations instead of mixing token and position embeddings at the input layer.

## Key Points

- [[ke-2021-rethinking-2006-15595]] defines TUPE-A with `α_ij = (x_i W^Q)(x_j W^K)^T / sqrt(2d) + (p_i U^Q)(p_j U^K)^T / sqrt(2d)`.
- TUPE-R extends the same decomposition by adding a relative bias term `b_{j-i}` inside the attention score.
- The method also resets position-only scores involving [[cls-token]] to learnable constants `θ_1` and `θ_2`, so sentence aggregation is less tied to the first positions.
- In BERT-Base, the extra positional projections add about `1.18M` parameters, roughly `1%` of the model, and the position correlation can be reused across layers.
- Empirically, TUPE improves GLUE accuracy and reaches baseline-level quality with much fewer pre-training steps.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[ke-2021-rethinking-2006-15595]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[ke-2021-rethinking-2006-15595]].
