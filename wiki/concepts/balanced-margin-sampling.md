---
type: concept
title: Balanced Margin Sampling
slug: balanced-margin-sampling
date: 2026-04-20
updated: 2026-04-20
aliases: [TAS-Balanced, 平衡边际采样]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Balanced Margin Sampling** (平衡边际采样) — a passage-pair sampling method that balances training examples across teacher-score margin ranges so a student model sees both easy and hard contrasts instead of being dominated by easy negatives.

## Key Points

- For each query, the paper partitions pairwise teacher margins into `10` bins that uniformly span the observed minimum-to-maximum margin range.
- Passage pairs are sampled by first drawing a margin bin and then drawing a positive-negative pair from that filtered subset.
- This reduces the skew toward high-margin pairs, which are abundant but low in information gain.
- The balanced margin stage is combined with topic-aware query sampling, yielding the full TAS-Balanced training recipe.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[hofst-tter-2021-efficiently-2104-06967]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[hofst-tter-2021-efficiently-2104-06967]].
