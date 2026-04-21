---
type: concept
title: Evolving Evaluation
slug: evolving-evaluation
date: 2026-04-20
updated: 2026-04-20
aliases: [rolling evaluation]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Evolving Evaluation** (演化式评测) — a benchmark design that repeatedly refreshes test data from recent real-world content to assess models on unseen and changing knowledge.

## Key Points

- KoLA updates its evolving benchmark season every `3` months and collects at least `500` newly published articles from roughly the previous `90` days.
- The evolving split is meant to reduce data leakage, test whether models can handle unseen knowledge, and expose hidden knowledge-updating mechanisms such as external search.
- KoLA reports that performance on evolving and non-evolving tasks remains linearly correlated, which the authors use as evidence that the evolving construction is reliable rather than arbitrary.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[yu-2024-kola-2306-09296]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[yu-2024-kola-2306-09296]].
