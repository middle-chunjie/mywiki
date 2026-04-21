---
type: concept
title: CLS Token
slug: cls-token
date: 2026-04-20
updated: 2026-04-20
aliases: ["[CLS]", classification token, 分类标记]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**CLS Token** (分类标记) — a special sequence-level token prepended to encoder inputs so its final hidden state can act as a sentence representation for downstream prediction.

## Key Points

- [[ke-2021-rethinking-2006-15595]] argues that `[CLS]` should not share the same positional treatment as ordinary lexical tokens.
- TUPE resets position-only correlations involving `[CLS]` to learnable values `θ_1` and `θ_2` rather than using the standard first-row and first-column positional scores.
- The motivation is that some attention heads naturally prefer local patterns, which can otherwise bias `[CLS]` toward early words instead of global sentence meaning.
- The ablation TUPE-A-tie-cls underperforms TUPE-A, especially on lower-resource sentence-level tasks such as CoLA and RTE.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[ke-2021-rethinking-2006-15595]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[ke-2021-rethinking-2006-15595]].
