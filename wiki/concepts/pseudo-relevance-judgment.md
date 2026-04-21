---
type: concept
title: Pseudo-Relevance Judgment
slug: pseudo-relevance-judgment
date: 2026-04-20
updated: 2026-04-20
aliases: [PRJ, 伪相关判断]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Pseudo-Relevance Judgment** (伪相关判断) — a heuristic labeling procedure that infers whether a context item is useful by checking whether adding it improves retrieval performance for the current query.

## Key Points

- In HAConvDR, a historical turn is marked relevant if retrieval with `` `q_n ∘ q_i ∘ p_i^*` `` beats retrieval with `` `q_n` `` alone under metric `` `M` ``.
- The method extends earlier query-only pseudo labeling by incorporating the historical gold passage, inspired by pseudo-relevance feedback.
- PRJ is reused twice: to choose history for denoised reformulation and to split historical passages into pseudo positives and historical hard negatives.
- The paper's deeper analysis shows PRJ marks only a minority of historical turns as relevant, especially highlighting the value of selective history usage.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[mo-2024-historyaware-2401-16659]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[mo-2024-historyaware-2401-16659]].
