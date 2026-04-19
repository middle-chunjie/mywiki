---
type: concept
title: Multi-Head Attention
slug: multi-head-attention
date: 2026-04-17
updated: 2026-04-17
aliases: [Multi-Head Attention, MHA, 多头注意力]
tags: [attention, nlp]
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-17
---

## Definition

Multi-Head Attention (多头注意力) — a mechanism that projects queries, keys, and values into `h` lower-dimensional subspaces via learned linear projections, runs [[scaled-dot-product-attention]] in parallel per head, concatenates the outputs, and projects back to the model dimension.

## Key Points

- `MultiHead(Q, K, V) = Concat(head_1, ..., head_h) Wᴼ`, where each `head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)`.
- Base Transformer configuration: `h = 8` heads with `d_k = d_v = d_model / h = 64`; total compute similar to a single full-dimensional head.
- Motivation: a single head's softmax averaging inhibits attending to multiple positions/subspaces simultaneously; heads specialize to different syntactic/semantic roles.
- Three usage modes in the [[transformer]]: encoder self-attention, masked decoder self-attention, encoder-decoder cross-attention.
- Too few heads (1) or too many (32) both hurt BLEU — quality peaks around 8–16 heads for the base model (Table 3, row A).

## My Position

<!-- User's stance on this concept. Fed by personal writing; tag "(personal stance)" on such bullets. -->

## Contradictions

<!-- Disagreements among sources about this concept. -->

## Sources

- [[vaswani-2017-attention-1706-03762]]

## Evolution Log

- 2026-04-17 (1 source): established from [[vaswani-2017-attention-1706-03762]] as parallel attention over projected subspaces (h=8, d_k=d_v=64).
