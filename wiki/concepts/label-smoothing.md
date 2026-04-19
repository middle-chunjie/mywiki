---
type: concept
title: Label Smoothing
slug: label-smoothing
date: 2026-04-17
updated: 2026-04-17
aliases: [Label Smoothing, 标签平滑]
tags: [regularization, training]
source_count: 1
confidence: low
domain_volatility: low
last_reviewed: 2026-04-17
---

## Definition

Label Smoothing (标签平滑) — a regularization technique that replaces hard one-hot targets with a mixture of the true label and a uniform distribution, discouraging the model from placing all probability mass on a single class.

## Key Points

- Used in [[vaswani-2017-attention-1706-03762]] with smoothing value `ε_ls = 0.1`.
- Trade-off: hurts perplexity (model is forced to stay "unsure") but improves accuracy and BLEU.
- Originates in Szegedy et al. (Rethinking the Inception Architecture, 2015); applied per-token in the Transformer's sequence output.
- One of three regularizers in the original Transformer training recipe, alongside residual dropout and the implicit regularization from attention + post-norm.

## My Position

<!-- User's stance on this concept. Fed by personal writing; tag "(personal stance)" on such bullets. -->

## Contradictions

<!-- Disagreements among sources about this concept. -->

## Sources

- [[vaswani-2017-attention-1706-03762]]

## Evolution Log

- 2026-04-17 (1 source): established from [[vaswani-2017-attention-1706-03762]] at `ε_ls = 0.1`, hurting perplexity but improving BLEU.
