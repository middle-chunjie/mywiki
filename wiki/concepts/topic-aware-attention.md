---
type: concept
title: Topic-aware Attention
slug: topic-aware-attention
date: 2026-04-20
updated: 2026-04-20
aliases: [主题感知注意力]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Topic-aware Attention** (主题感知注意力) — an attention mechanism that reweights token representations according to their relevance to global topic signals such as titles or topic summaries.

## Key Points

- In [[fang-2021-guided]], topic-aware attention uses the title-side summary vector `q_t` as a query to score document tokens.
- The token representation is updated as `u_i = α_i h_i`, with `α_i = SoftMax(v_1^T tanh(W_1 h_i + W_2 q_t))`.
- Its purpose is to highlight topic-related words that are more likely to be concepts under the document's global semantics.
- Ablation shows this module mainly helps recall: removing it lowers recall across CSEN, KP-20K, and MTB.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[fang-2021-guided]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[fang-2021-guided]].
