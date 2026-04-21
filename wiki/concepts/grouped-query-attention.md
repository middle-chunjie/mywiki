---
type: concept
title: Grouped-Query Attention
slug: grouped-query-attention
date: 2026-04-20
updated: 2026-04-20
aliases: [GQA, grouped query attention]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Grouped-Query Attention** — an attention variant in which multiple query heads share grouped key and value projections, reducing key-value cache cost relative to full multi-head attention.

## Key Points

- DeepSeek-V2 uses GQA as a comparison point when motivating MLA's cache-efficiency design.
- The paper characterizes GQA as a compromise that reduces KV cache more than MHA but still gives up some performance.
- Table 1 compares GQA's cache as `2 n_g d_h l`, where `n_g` is the number of key-value groups, against MLA's smaller `(d_c + d_h^R) l`.
- The authors frame MLA as targeting the same inference bottleneck as GQA while avoiding the capability drop they attribute to grouped KV sharing.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[deepseek-ai-2024-deepseekv-2405-04434]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[deepseek-ai-2024-deepseekv-2405-04434]].
