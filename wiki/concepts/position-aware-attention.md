---
type: concept
title: Position-aware Attention
slug: position-aware-attention
date: 2026-04-20
updated: 2026-04-20
aliases: [位置感知注意力]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Position-aware Attention** (位置感知注意力) — an attention mechanism that incorporates relative position information so token importance depends on distance to a salient span or anchor.

## Key Points

- [[fang-2021-guided]] defines relative position `p_i` from each token to the matched clue-word span `(s_1, s_2)` and embeds it with a shared position matrix.
- The clue-word hidden states are averaged into query `q`, then attention weights are computed with token, query, and position features before forming `m_i = β_i h_i`.
- This module explicitly models the local positional and semantic relationship between clue words and nearby concept words.
- Removing it lowers precision more strongly than recall, suggesting clue-word-centered local context is especially useful for precise boundary detection.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[fang-2021-guided]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[fang-2021-guided]].
