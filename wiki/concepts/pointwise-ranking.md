---
type: concept
title: Pointwise Ranking
slug: pointwise-ranking
date: 2026-04-20
updated: 2026-04-20
aliases: [point-wise ranking, 点式排序]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Pointwise Ranking** (点式排序) — a ranking formulation that assigns an independent relevance score to each query-document pair and sorts documents by those scores.

## Key Points

- The paper analyzes relevance generation and query generation as representative pointwise LLM ranking methods.
- Pointwise prompting requires calibrated scores that are comparable across different documents, which the paper argues is difficult and unnecessary for ranking.
- One analyzed relevance-generation score is `s_i = 1 + p(Yes)` when the model outputs `Yes`, and `s_i = 1 - p(No)` when it outputs `No`.
- Pointwise methods depend on access to token probabilities, so they do not naturally work with generation-only APIs such as GPT-4-style interfaces.
- The paper positions PRP as a better match for relative ordering because ranking only needs pairwise preference, not absolute calibration.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[qin-2024-large-2306-17563]]
- [[ma-2023-finetuning-2310-08319]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[qin-2024-large-2306-17563]].
