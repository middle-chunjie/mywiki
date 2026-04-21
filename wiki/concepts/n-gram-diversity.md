---
type: concept
title: N-gram Diversity
slug: n-gram-diversity
date: 2026-04-20
updated: 2026-04-20
aliases: [distinct n-grams, N 元组多样性]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**N-gram Diversity** (N 元组多样性) — a text-diversity metric defined by the proportion of unique n-grams among all n-grams in a generated sample.

## Key Points

- The paper treats n-gram diversity as a standard text-generation baseline for evaluating diverse decoding.
- It shows that n-gram diversity is strongly correlated with Vendi Score when both use the same n-gram-based representation, but the rankings can still differ.
- The paper highlights two failure modes: repeated phrases within one sentence depress n-gram diversity, while one unusually long sentence can inflate it.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[friedman-2023-vendi-2210-02410]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[friedman-2023-vendi-2210-02410]].
