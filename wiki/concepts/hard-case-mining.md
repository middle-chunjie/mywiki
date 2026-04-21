---
type: concept
title: Hard-Case Mining
slug: hard-case-mining
date: 2026-04-20
updated: 2026-04-20
aliases: [hard case mining, failure case mining, 困难案例挖掘]
tags: [agents, training, self-evolving]
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Hard-Case Mining** (困难案例挖掘) — the process of automatically identifying and prioritizing the most informative failure cases from recent training steps to drive targeted self-improvement of a model or skill set.

## Key Points

- MemSkill maintains a sliding-window hard-case buffer; cases expire if they exceed a maximum training-step gap or if the buffer reaches capacity, preventing unbounded growth.
- Each case is scored by `d(q) = (1 - r(q)) · c(q)`, where `r(q)` is task reward and `c(q)` is the cumulative failure count, so low-reward repeatedly-failed cases are prioritized.
- Representative cases are mined via KMeans clustering over query semantic similarity, ensuring the designer receives *diverse* failure signals rather than a single dominant error mode.
- The designer uses these representative hard cases to analyze missing or mis-specified memory behaviors, proposing skill refinements or new skills in a two-stage LLM process.
- This approach is analogous to curriculum learning or hard-negative mining in retrieval; the main difference is that the mined cases drive *operation design* rather than data weighting.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[zhang-2026-memskill-2602-02474]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[zhang-2026-memskill-2602-02474]].
