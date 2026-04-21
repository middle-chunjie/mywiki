---
type: concept
title: Catastrophic Forgetting
slug: catastrophic-forgetting
date: 2026-04-20
updated: 2026-04-20
aliases: [灾难性遗忘]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Catastrophic Forgetting** (灾难性遗忘) — the failure mode where a model loses performance on earlier tasks after being updated on later tasks.

## Key Points

- The paper uses catastrophic forgetting as the key behavioral contrast between local spline-based KANs and globally activated MLPs.
- In its five-peak sequential regression toy task, MLPs reshape the whole function after each new phase, erasing earlier peaks.
- KANs largely restrict updates to the currently observed region because spline bases are local, so previously learned regions remain stable.
- The paper does not claim the problem is solved in general; it only demonstrates a promising locality mechanism in a simple low-dimensional setting.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[liu-2024-kan-2404-19756]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[liu-2024-kan-2404-19756]].
