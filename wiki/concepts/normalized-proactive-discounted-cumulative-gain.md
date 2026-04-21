---
type: concept
title: Normalized Proactive Discounted Cumulative Gain
slug: normalized-proactive-discounted-cumulative-gain
date: 2026-04-20
updated: 2026-04-20
aliases:
  - npDCG
  - 归一化主动折损累计增益
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Normalized Proactive Discounted Cumulative Gain** (归一化主动折损累计增益) — a proactive retrieval metric that scores ranked interventions by document relevance, intervention timing, and normalization against an ideal proactive policy.

## Key Points

- `npDCG` extends ranking evaluation to settings where systems can intervene at multiple conversation turns.
- The metric removes duplicate documents already shown in earlier turns so repeated suggestions do not gain extra credit.
- Relevance is down-weighted when a useful document is surfaced after its ideal first helpful utterance.
- The final score is `pDCG / ipDCG`, making values comparable across conversations with different numbers of opportunities.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[samarinas-2024-procis]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[samarinas-2024-procis]].
