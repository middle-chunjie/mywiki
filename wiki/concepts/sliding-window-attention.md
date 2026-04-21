---
type: concept
title: Sliding-Window Attention
slug: sliding-window-attention
date: 2026-04-20
updated: 2026-04-20
aliases: [window attention, 滑动窗口注意力]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Sliding-Window Attention** (滑动窗口注意力) — an attention pattern in which each token can only attend to a bounded local window of neighboring tokens, trading global coverage for linear-time scaling.

## Key Points

- LongCoder uses a causal left window where token `i` can attend only to positions `j` satisfying `i - j <= w`.
- The paper sets `w = 512` during sparse-model training and evaluation to match the local context available to dense `512`-token baselines.
- Empirically, the authors report that a window of `256` already covers more than `90%` of average attention mass in CodeGPT.
- Window attention alone preserves fast inference, but by itself it still requires many hops to retrieve distant dependencies.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[guo-2023-longcoder-2306-14893]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[guo-2023-longcoder-2306-14893]].
