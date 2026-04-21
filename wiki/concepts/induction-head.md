---
type: concept
title: Induction Head
slug: induction-head
date: 2026-04-20
updated: 2026-04-20
aliases: [归纳头]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Induction Head** (归纳头) — an attention-based circuit that detects a previously seen token pattern and predicts its continuation when the prefix reappears later in the sequence.

## Key Points

- The paper uses synthetic children-story data with two-token names to measure induction capability in controlled settings.
- For models of `30M` parameters and below, 2-token prediction greatly improves induction accuracy relative to next-token prediction.
- The advantage largely disappears by `100M` parameters, suggesting MTP mainly helps induction features emerge earlier rather than changing the final asymptotic capability.
- A higher-quality data mixture can cause induction capability to form earlier even without MTP, shrinking the gap between losses.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[gloeckle-2024-better-2404-19737]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[gloeckle-2024-better-2404-19737]].
