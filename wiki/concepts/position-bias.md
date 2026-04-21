---
type: concept
title: Position Bias
slug: position-bias
date: 2026-04-20
updated: 2026-04-20
aliases: [position bias, 位置偏置]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Position Bias** (位置偏置) — a learned tendency to treat some token positions as more informative than others regardless of the actual content they contain.

## Key Points

- The paper hypothesizes that autoregressive pretraining favors nearby prefix tokens and that instruction-following data often places crucial control information near the beginning of the prompt.
- Under this view, long-context failures are not only a matter of window size but also of biased supervision about where important information usually appears.
- Information-intensive training counters this bias by randomizing the positions of answer-bearing segments, and the resulting robustness improvements provide indirect support for the hypothesis.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[an-2024-make-2404-16811]]
- [[unknown-nd-evaluating-2310-07641]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[an-2024-make-2404-16811]].
