---
type: concept
title: Prompt Retrieval
slug: prompt-retrieval
date: 2026-04-20
updated: 2026-04-20
aliases: [example retrieval for ICL, in-context example selection, 提示检索]
tags: [retrieval, in-context-learning, prompting]
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Prompt Retrieval** (提示检索) — a family of methods that automatically select the most suitable in-context examples from a source dataset for a given query, replacing random selection with a learned or heuristic scoring function.

## Key Points

- The core objective is `x* = argmax_{x_n ∈ D} f_θ(x_n, x_q)` where `f_θ` scores suitability of a candidate `x_n` for query `x_q`.
- Unsupervised prompt retrieval uses a frozen off-the-shelf encoder (e.g., CLIP) to compute cosine similarity; no training required.
- Supervised prompt retrieval fine-tunes the encoder via contrastive learning to directly optimize downstream in-context learning performance.
- Both methods outperform random selection on visual tasks; the supervised approach yields best results (~+8% mIoU vs. random on Pascal-5^i segmentation).
- Inspired by analogous work in NLP (Liu et al., 2021; Rubin et al., 2021) which showed that semantically similar examples benefit language model ICL.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[zhang-nd-what]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[zhang-nd-what]].
