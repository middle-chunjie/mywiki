---
type: concept
title: Supervised Positive Inclusion
slug: supervised-positive-inclusion
date: 2026-04-20
updated: 2026-04-20
aliases: [SPI, 监督正例纳入]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Supervised Positive Inclusion** (监督正例纳入) — a contrastive-learning strategy that enlarges the positive set for an anchor using label-confirmed positive examples rather than relying on a single augmented counterpart.

## Key Points

- CL4CVR adds extra positives only when the anchor has conversion label `1`, using scarce but reliable conversion events to enrich the contrastive signal.
- The positive set is `S(i) = {j} ∪ {k | z(ẽ_k) = z(ẽ_i) = 1, k ≠ i, k ≠ j}`.
- The method intentionally avoids adding label-0 examples as positives because abundant negatives would collapse contrast in sparse-conversion batches.
- EM + SPI improves over EM alone on both datasets, and the full EM + FNE + SPI system performs best overall.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[ouyang-2023-contrastive]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[ouyang-2023-contrastive]].
