---
type: concept
title: N-gram Language Model
slug: n-gram-language-model
date: 2026-04-20
updated: 2026-04-20
aliases: [ngram language model, ngram lm, n元语言模型]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**N-gram Language Model** (n元语言模型) — a count-based language model that estimates the next-token distribution from the frequency of a fixed-length context and continuation in a reference corpus.

## Key Points

- The paper argues that prior `n`-gram LMs were underestimated partly because they were constrained to small fixed orders such as `n <= 5`.
- Its baseline probability is the standard count ratio `P_n(w_i | context) = cnt(context, w_i) / cnt(context)`.
- At trillion-token scale, fixed-order `5`-grams still underperform because they discard useful long-context evidence.
- The paper modernizes the paradigm by replacing precomputed count tables with a suffix-array index and by extending fixed-order `n`-grams into an unbounded `∞`-gram variant.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[liu-2024-infinigram-2401-17377]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[liu-2024-infinigram-2401-17377]].
