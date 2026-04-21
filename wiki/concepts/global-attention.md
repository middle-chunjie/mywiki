---
type: concept
title: Global Attention
slug: global-attention
date: 2026-04-20
updated: 2026-04-20
aliases: [全局注意力]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Global Attention** (全局注意力) — an attention pattern that makes a selected subset of tokens accessible from the full sequence so that information with broad scope can be retrieved directly.

## Key Points

- LongCoder marks line-feed positions corresponding to imports and class/function/structure definitions as globally accessible positions `G`.
- The selected positions are obtained from syntax analysis with `tree-sitter` rather than learned purely from attention weights.
- Because only `k << n` positions receive global access, the added complexity remains approximately linear in sequence length.
- The paper motivates this design from code semantics: imported packages and definitions often affect completion far beyond the local window.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[guo-2023-longcoder-2306-14893]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[guo-2023-longcoder-2306-14893]].
