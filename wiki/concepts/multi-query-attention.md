---
type: concept
title: Multi-Query Attention
slug: multi-query-attention
date: 2026-04-20
updated: 2026-04-20
aliases: [MQA]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Multi-query attention** — an attention variant where multiple query heads share a single key projection and a single value projection, reducing autoregressive decoding cost and key-value cache size while preserving most quality.

## Key Points

- PaLM keeps per-head query projections but shares key and value projections across heads to improve decoding efficiency.
- The paper reports this change as roughly quality-neutral during training while yielding significant inference-time savings.
- The design targets the hardware inefficiency of standard multi-head attention during one-token-at-a-time autoregressive decoding.
- In PaLM, multi-query attention is paired with a decoder-only Transformer, RoPE, and large-scale few-shot evaluation.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[chowdhery-2022-palm-2204-02311]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[chowdhery-2022-palm-2204-02311]].
