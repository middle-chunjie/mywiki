---
type: concept
title: Pairwise Ranking Prompting
slug: pairwise-ranking-prompting
date: 2026-04-20
updated: 2026-04-20
aliases: [PRP, pairwise ranking prompting, 成对排序提示]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Pairwise Ranking Prompting** (成对排序提示) — a ranking formulation that asks an LLM to compare two candidate documents for the same query and then aggregates those local preferences into a global ranking.

## Key Points

- The paper uses a basic comparison unit `u(q, d_1, d_2)` instead of asking for calibrated scores or a full permutation over many documents.
- PRP supports both scoring APIs and generation APIs because the output space is reduced to two choices, `"Passage A"` or `"Passage B"`.
- The method queries both document orders to reduce prompt-order bias and treats inconsistent decisions as ties.
- The paper instantiates PRP with three aggregation strategies: all-pair comparison, sorting-based comparison, and sliding-window refinement.
- PRP is the paper's core explanation for why moderate-sized open LLMs can become competitive zero-shot rerankers.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[qin-2024-large-2306-17563]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[qin-2024-large-2306-17563]].
