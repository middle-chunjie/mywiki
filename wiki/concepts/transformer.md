---
type: concept
title: Transformer
slug: transformer
date: 2026-04-17
updated: 2026-04-17
aliases: [Transformer, 变换器, Transformer架构]
tags: [architecture, nlp, llm]
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-17
---

## Definition

Transformer (变换器) — an encoder–decoder sequence transduction architecture that replaces recurrence and convolution entirely with stacked multi-head self-attention and position-wise feed-forward layers, connected by residual connections and layer normalization.

## Key Points

- Introduced in [[vaswani-2017-attention-1706-03762]] as a solution to the parallelism bottleneck of RNN-based seq2seq.
- Standard base configuration: `N = 6` encoder and decoder layers, `d_model = 512`, `d_ff = 2048`, `h = 8` heads.
- Attends globally with constant path length between any two positions (`O(1)` vs. RNN's `O(n)`).
- Self-attention cost per layer is `O(n²·d)`, faster than RNNs when `n < d` but quadratic in sequence length.
- Decoder uses masked self-attention plus cross-attention over encoder outputs to preserve auto-regression.
- Architectural foundation for essentially all subsequent large language models.

## My Position

<!-- User's stance on this concept. Fed by personal writing; tag "(personal stance)" on such bullets. -->

## Contradictions

<!-- Disagreements among sources about this concept. -->

## Sources

- [[vaswani-2017-attention-1706-03762]]

## Evolution Log

- 2026-04-17 (1 source): established from [[vaswani-2017-attention-1706-03762]] as attention-only encoder-decoder architecture.
