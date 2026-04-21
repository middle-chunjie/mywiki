---
type: concept
title: Sliding Window
slug: sliding-window
date: 2026-04-20
updated: 2026-04-20
aliases: [sliding window, 滑动窗口]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Sliding Window** (滑动窗口) — a procedure that processes a long ranked list through overlapping local windows so a model with limited context can iteratively update the ordering.

## Key Points

- This paper applies sliding-window re-ranking because ChatGPT and GPT-4 cannot jointly process all top-`100` BM25 passages in one prompt.
- The default configuration is window size `20` and step size `10`, applied in back-to-front order over the candidate list.
- Overlap between adjacent windows lets high-ranked passages from one window compete again in the next window instead of being fixed after one pass.
- Ablation shows performance is sensitive to both the initial order and the number of passes; random initial order collapses TREC-DL19 nDCG@10 to `25.17`.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[sun-2023-chatgpt-2304-09542]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[sun-2023-chatgpt-2304-09542]].
