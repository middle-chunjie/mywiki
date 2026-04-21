---
type: concept
title: Synthetic Recall
slug: synthetic-recall
date: 2026-04-20
updated: 2026-04-20
aliases: [synthetic-recall-task]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Synthetic Recall** (合成召回任务) — procedurally constructed long-context tests that measure whether a model can retrieve target information embedded in controlled distractor contexts.

## Key Points

- HELMET keeps synthetic recall as one benchmark category but explicitly argues it should not be the only signal for long-context capability.
- The paper includes JSON KV and selected RULER recall tasks rather than relying on simple needle-in-a-haystack alone.
- Harder synthetic recall settings with more distracting contexts correlate better with downstream tasks than saturated easy variants.
- Across `35` instruction-tuned models at `128K`, no synthetic task achieves average Spearman correlation above `0.8` with real-world downstream performance.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[yen-2024-helmet-2410-02694]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[yen-2024-helmet-2410-02694]].
