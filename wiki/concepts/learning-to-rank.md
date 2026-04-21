---
type: concept
title: Learning to Rank
slug: learning-to-rank
date: 2026-04-20
updated: 2026-04-20
aliases: [学习排序, LTR]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Learning to Rank** (学习排序) — a family of methods that optimize the ordering of candidates so more relevant items are scored above less relevant ones.

## Key Points

- The paper applies learning-to-rank to order patents for each plaintiff-defendant pair by litigation risk.
- Training uses pairwise preferences between observed litigated patents and sampled unlitigated patents.
- The probabilistic ranking objective is implemented with a sigmoid over score differences.
- Ranking is chosen because the operational goal is to return a shortlist of risky patents rather than only a yes-or-no decision.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[liu-2018-patent]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[liu-2018-patent]].
