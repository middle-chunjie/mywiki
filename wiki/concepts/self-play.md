---
type: concept
title: Self-Play
slug: self-play
date: 2026-04-20
updated: 2026-04-20
aliases: [synthetic self-play]
tags: [llm, data-generation, code]
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Self-Play** — a data-generation procedure in which a model creates new tasks or candidate solutions that are then reused to train or improve another model on the same domain.

## Key Points

- The paper uses GPT-3.5 to generate novel competitive-programming problems and then optimize them with a PIE-trained model.
- From `10,000` generation attempts, the pipeline keeps `6,553` programs outside the original splits and groups them into `3,314` equivalence sets.
- After filtering for at least `5x` speedup and capping semantic duplicates at `3`, the method yields `1,485` optimized synthetic examples.
- Adding self-play data modestly improves generalization, especially for GPT-3.5 and `Best@1` optimization coverage.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[shypula-2024-performanceimproving-2302-07867]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[shypula-2024-performanceimproving-2302-07867]].
