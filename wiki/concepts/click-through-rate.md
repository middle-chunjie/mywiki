---
type: concept
title: Click-Through Rate
slug: click-through-rate
date: 2026-04-20
updated: 2026-04-20
aliases: [CTR, click through rate, 点击率]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Click-Through Rate** (点击率) — the proportion of displayed items that receive a click, commonly used as an engagement signal for ranking or recommendation quality.

## Key Points

- CTR is the main feedback signal used to refine the generated question pool across iterations.
- The simulator derives question-level click probabilities from persona-specific relevance scores via a softmax model with rejection option.
- The method uses CTR both to drop the worst question in the item pool and to condition future generations in the optimization prompt.
- Reported gains in CTR are the main evidence that Explore-Exploit finds questions better aligned with hidden user preferences.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[senel-2024-generative-2406-05255]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[senel-2024-generative-2406-05255]].
